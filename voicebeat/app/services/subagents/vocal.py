"""
Vocal Subagent — LLM-designed vocal processing.

Melodic mode: LLM assigns MIDI notes to words, processor pitch-shifts.
Rhythmic mode: LLM assigns beat positions to words, processor time-snaps.
"""

import json
import uuid
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.models.schemas import Layer, Project, Instrument, SegmentType
from app.services import vocal_processor
from app.services.subagents.base import BaseSubagent

logger = logging.getLogger(__name__)


# ── Tool definitions ──────────────────────────────────────────────────

TOOL_DESIGN_MELODY = {
    "type": "function",
    "function": {
        "name": "design_vocal_melody",
        "description": (
            "Assign a MIDI note and timing to each word to create a vocal melody. "
            "Each word is synthesized via TTS then pitch-shifted. "
            "IMPORTANT: Use whole words only — never split into syllables (no 'Mu-' + '-sic', use 'Music')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "description": "A complete word — never split into syllables"},
                            "midi_note": {
                                "type": "integer",
                                "description": "Target MIDI note (60=C4, 62=D4, 64=E4, etc.)",
                            },
                            "start_beat": {
                                "type": "number",
                                "description": "Beat position to start this word (0-based, quarter note beats)",
                            },
                            "duration_beats": {
                                "type": "number",
                                "description": "How long the word should last in beats",
                            },
                        },
                        "required": ["word", "midi_note", "start_beat", "duration_beats"],
                    },
                },
            },
            "required": ["notes"],
        },
    },
}

TOOL_DESIGN_RHYTHM = {
    "type": "function",
    "function": {
        "name": "design_vocal_rhythm",
        "description": (
            "Assign beat-grid positions to each word for rhythmic vocal placement. "
            "The user's speech will be time-stretched and repositioned to snap to these positions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "placements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "description": "A complete word — never split into syllables"},
                            "bar": {"type": "integer", "description": "Which bar (0-indexed)"},
                            "beat_position": {
                                "type": "integer",
                                "description": "Position on 16th-note grid (0-15) within the bar",
                            },
                        },
                        "required": ["word", "bar", "beat_position"],
                    },
                },
            },
            "required": ["placements"],
        },
    },
}

TOOL_RENDER_VOCAL = {
    "type": "function",
    "function": {
        "name": "render_vocal",
        "description": "Render the designed vocal — applies pitch-shifts or time-stretches to produce audio.",
        "parameters": {
            "type": "object",
            "properties": {
                "layer_name": {
                    "type": "string",
                    "description": "Name for the vocal layer (e.g. 'Lead Vocals', 'Rap Flow')",
                },
            },
            "required": ["layer_name"],
        },
    },
}

TOOL_DONE = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "Signal that vocal processing is complete.",
        "parameters": {"type": "object", "properties": {}},
    },
}


# ── Vocal Subagent ────────────────────────────────────────────────────

class VocalSubagent(BaseSubagent):
    """LLM-backed vocal processor with melodic and rhythmic modes."""

    name = "vocal"
    max_iterations = 4

    def __init__(self, mode: str = "melodic"):
        self.mode = mode  # "melodic" or "rhythmic"
        self._design: list[dict] | None = None

    def get_system_prompt(self, project: Project, instructions: dict) -> str:
        key = project.key_signature or "C major"
        bpm = project.bpm

        lyrics_filter = (
            "## IMPORTANT: Lyrics filtering\n"
            "The word list may include the user's full speech transcript with instructions mixed in "
            "(e.g. 'make me a rock song about...'). ONLY include actual lyric words in your design — "
            "skip instruction/direction words. If the transcript is entirely instructional, compose "
            "suitable lyrics based on the described theme."
        )

        if self.mode == "melodic":
            return f"""You are a vocal melody designer. You receive a word list (possibly from a speech transcript)
and design a singing melody for the lyrics.

## Your task
Assign a MIDI note, start beat, and duration to each lyric word using design_vocal_melody.
Then call render_vocal to produce the audio. Then call done.

## Musical context
- Key: {key}
- BPM: {bpm}
- Scale notes for {key}: use notes that fit this key

{lyrics_filter}

## Guidelines
- NEVER split words into syllables — each entry must be a complete word (use "music" not "mu-" + "-sic")
- Keep the melody simple and singable — stepwise motion, small intervals
- Keep pitch shifts small — stay within ±4 semitones of a natural speaking range (MIDI 58-67)
- Match natural speech emphasis: stressed syllables get higher notes or longer durations
- Short words (articles, prepositions) can share a note with adjacent words
- Place words on or near beat boundaries for a natural feel
- Leave small gaps between phrases for breathing"""

        else:  # rhythmic
            return f"""You are a vocal rhythm designer for rap/spoken word. You receive a word list
(possibly from a speech transcript) and design beat-grid placements for the lyrics.

## Your task
Assign a bar and beat_position (0-15 on the 16th-note grid) to each lyric word using design_vocal_rhythm.
Then call render_vocal to produce the audio. Then call done.

## Musical context
- BPM: {bpm}
- 16th-note grid: positions 0-15 per bar, where 0=downbeat, 4=beat 2, 8=beat 3, 12=beat 4

{lyrics_filter}

## Guidelines
- Emphasize the groove — key words on strong beats (0, 4, 8, 12)
- Pack syllables tightly for fast flow, space them for emphasis
- Leave position 0 of bar 0 for a strong opening word
- End phrases before the next bar for breathing room
- Rhyming words should land on similar beat positions for rhythmic consistency"""

    def get_tools(self) -> list[dict]:
        if self.mode == "melodic":
            return [TOOL_DESIGN_MELODY, TOOL_RENDER_VOCAL, TOOL_DONE]
        else:
            return [TOOL_DESIGN_RHYTHM, TOOL_RENDER_VOCAL, TOOL_DONE]

    def build_user_message(self, project: Project, instructions: dict) -> str:
        words = instructions.get("words", [])
        lines = [f"# Vocal Assignment\n"]
        lines.append(f"- BPM: {project.bpm}")
        lines.append(f"- Key: {project.key_signature or 'C major'}")
        lines.append(f"- Genre: {instructions.get('genre', 'pop')}")
        lines.append(f"- Mode: {self.mode}\n")

        if instructions.get("style_notes"):
            lines.append(f"## Style: {instructions['style_notes']}\n")

        if instructions.get("transcript"):
            lines.append(f'## Full transcript:\n"{instructions["transcript"]}"\n')

        lines.append("## Words (select only lyrics, skip instructions):")
        for i, w in enumerate(words, 1):
            if "start" in w and "end" in w:
                dur = w["end"] - w["start"]
                lines.append(f'{i}. "{w["word"]}" — {w["start"]:.2f}s–{w["end"]:.2f}s ({dur:.2f}s)')
            else:
                lines.append(f'{i}. "{w["word"]}"')

        if instructions.get("sections"):
            total_bars = max((s.get("end_bar", 0) for s in instructions["sections"]), default=4)
            lines.append(f"\n## Target: fit lyrics into {total_bars} bars")

        lines.append(f"\nDesign the vocal {'melody' if self.mode == 'melodic' else 'rhythm'} now.")
        lines.append("Call the design tool, then render_vocal, then done.")
        return "\n".join(lines)

    async def execute_tool(
        self,
        tool_call,
        project: Project,
        instructions: dict,
        created_layers: list[Layer],
    ) -> dict:
        fname = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if fname == "design_vocal_melody":
            self._design = args.get("notes", [])
            return {
                "success": True,
                "message": f"Melody designed for {len(self._design)} words",
            }

        elif fname == "design_vocal_rhythm":
            self._design = args.get("placements", [])
            return {
                "success": True,
                "message": f"Rhythm designed for {len(self._design)} words",
            }

        elif fname == "render_vocal":
            if self._design is None:
                return {"success": False, "error": "No design yet — call design tool first"}

            layer_name = args.get("layer_name", "Vocals")

            try:
                if self.mode == "melodic":
                    audio_file = await vocal_processor.tts_melodic_vocal(
                        melody_design=self._design,
                        bpm=project.bpm,
                    )
                else:
                    total_bars = 4
                    if instructions.get("sections"):
                        total_bars = max(
                            (s.get("end_bar", 0) for s in instructions["sections"]),
                            default=4,
                        )
                    audio_file = await vocal_processor.tts_rhythmic_vocal(
                        rhythm_design=self._design,
                        bpm=project.bpm,
                        total_bars=total_bars,
                    )
            except Exception as e:
                logger.error(f"VOCAL: Render failed: {e}")
                return {"success": False, "error": f"Render failed: {e}"}

            layer = Layer(
                id=str(uuid.uuid4()),
                name=layer_name,
                instrument=Instrument.VOCAL,
                segment_type=SegmentType.VOCAL,
                audio_file=audio_file,
            )
            created_layers.append(layer)

            return {
                "success": True,
                "message": f"Vocal rendered: '{layer_name}' -> {audio_file}",
                "layer_id": layer.id,
            }

        return {"success": False, "error": f"Unknown tool: {fname}"}
