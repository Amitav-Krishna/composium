"""
Base Subagent — shared LLM loop for all instrument subagents.

Each subagent inherits BaseSubagent, provides its own system prompt,
tool definitions, and tool execution logic.  The run() method drives
the OpenAI function-calling loop and returns the layers produced.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from openai import AsyncOpenAI
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import settings
from app.models.schemas import (
    Layer, Project, Instrument, QuantizedBeat, PitchEvent,
)

logger = logging.getLogger(__name__)


def _get_client() -> AsyncOpenAI:
    """Lazy-initialise the shared OpenAI client (GPT-4o, used by director)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _get_subagent_client() -> AsyncOpenAI:
    """Lazy-initialise the OpenRouter client (used by subagents for cheaper models)."""
    global _subagent_client
    if _subagent_client is None:
        _subagent_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
    return _subagent_client


_client: AsyncOpenAI | None = None
_subagent_client: AsyncOpenAI | None = None


# ---------------------------------------------------------------------------
# Shared helpers (extracted from the old monolithic agent)
# ---------------------------------------------------------------------------

def audio_duration(path: str) -> float:
    """Return audio file duration in seconds, or 0.0 on error."""
    try:
        from pydub import AudioSegment as PydubSegment
        return len(PydubSegment.from_file(path)) / 1000.0
    except Exception:
        return 0.0


def resolve_segment_id(short_id: str, segment_data: dict[str, dict]) -> str | None:
    """Resolve a possibly-truncated segment ID to the full ID."""
    if short_id in segment_data:
        return short_id
    for full_id in segment_data:
        if full_id.startswith(short_id):
            return full_id
    return None


def build_beats(beats_data: list[dict]) -> list[QuantizedBeat]:
    """Convert raw beat dicts from the LLM into QuantizedBeat objects."""
    beats = []
    for b in beats_data:
        try:
            inst = Instrument(b.get("instrument", "kick"))
        except ValueError:
            inst = Instrument.KICK
        beats.append(QuantizedBeat(
            position=b.get("position", 0),
            bar=b.get("bar", 0),
            instrument=inst,
            velocity=b.get("velocity", 1.0),
        ))
    return beats


def build_pitches(notes_data: list[dict], bpm: int) -> list[PitchEvent]:
    """Convert raw note dicts from the LLM into PitchEvent objects."""
    spb = 60.0 / bpm
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitches = []
    for n in notes_data:
        midi = n["midi_note"]
        start_beat = n["start_beat"]
        dur_beats = n["duration_beats"]
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        octave = (midi // 12) - 1
        note_name = f"{note_names[midi % 12]}{octave}"
        pitches.append(PitchEvent(
            time_seconds=start_beat * spb,
            frequency_hz=freq,
            midi_note=midi,
            note_name=note_name,
            duration_seconds=dur_beats * spb,
        ))
    return pitches


# ---------------------------------------------------------------------------
# BaseSubagent
# ---------------------------------------------------------------------------

class BaseSubagent(ABC):
    """Abstract base for LLM-backed instrument subagents."""

    name: str = "subagent"
    max_iterations: int = 6

    @abstractmethod
    def get_system_prompt(self, project: Project, instructions: dict) -> str:
        """Return the system prompt for this subagent."""
        ...

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return OpenAI tool definitions for this subagent."""
        ...

    @abstractmethod
    async def execute_tool(
        self,
        tool_call,
        project: Project,
        instructions: dict,
        created_layers: list[Layer],
    ) -> dict:
        """Execute a single tool call. Return a result dict."""
        ...

    def build_user_message(self, project: Project, instructions: dict) -> str:
        """Build the initial user message with context for the subagent."""
        lines = [f"# Assignment from Director\n"]

        lines.append(f"- BPM: {project.bpm}")
        lines.append(f"- Key: {project.key_signature or 'C major'}")
        if instructions.get("genre"):
            lines.append(f"- Genre: {instructions['genre']}")
        lines.append("")

        if instructions.get("sections"):
            total_bars = max((sec.get("end_bar", 0) for sec in instructions["sections"]), default=0)
            lines.append(f"## Sections to cover (total song: {total_bars} bars):")
            for sec in instructions["sections"]:
                bar_count = sec.get("end_bar", 0) - sec.get("start_bar", 0)
                lines.append(f"- {sec['name']}: bars {sec['start_bar']}–{sec['end_bar']} ({bar_count} bars) — {sec.get('description', '')}")
            lines.append("")

        if instructions.get("style_notes"):
            lines.append(f"## Style notes:\n{instructions['style_notes']}\n")

        if instructions.get("segment_id"):
            lines.append(f"## User audio segment: {instructions['segment_id']}")
            lines.append("Use render tools to incorporate the user's audio.\n")

        lines.append("Create the parts now. Call `done` when finished.")
        return "\n".join(lines)

    async def run(self, project: Project, instructions: dict) -> list[Layer]:
        """
        Run the LLM loop. Returns layers created by this subagent.

        Args:
            project: The shared project (read BPM/key, do NOT append layers here)
            instructions: Delegation dict from the director
        """
        log_prefix = self.name.upper()
        logger.info(f"{log_prefix}: Starting (max {self.max_iterations} iterations)")

        messages = [
            {"role": "system", "content": self.get_system_prompt(project, instructions)},
            {"role": "user", "content": self.build_user_message(project, instructions)},
        ]
        tools = self.get_tools()
        created_layers: list[Layer] = []

        for iteration in range(self.max_iterations):
            logger.info(f"{log_prefix}: Iteration {iteration + 1}")
            response = await _get_subagent_client().chat.completions.create(
                model=settings.subagent_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(msg)

            if msg.content:
                logger.info(f"{log_prefix}: {msg.content}")

            if not msg.tool_calls:
                logger.info(f"{log_prefix}: No tool calls, ending")
                break

            done_signaled = False
            for tc in msg.tool_calls:
                fname = tc.function.name
                logger.info(f"{log_prefix}: Tool call: {fname}({tc.function.arguments})")

                if fname == "done":
                    done_signaled = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"success": True}),
                    })
                    continue

                result = await self.execute_tool(tc, project, instructions, created_layers)
                logger.info(f"{log_prefix}: Result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            if done_signaled:
                logger.info(f"{log_prefix}: Done signal — produced {len(created_layers)} layers")
                return created_layers

        logger.info(f"{log_prefix}: Loop finished — produced {len(created_layers)} layers")
        return created_layers
