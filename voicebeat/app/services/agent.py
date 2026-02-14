"""
Agentic Orchestrator Service

Uses OpenAI with tool/function calling to interpret user instructions
and orchestrate the music creation pipeline.

Key insight from Orca: LLMs can't directly edit audio. Give them tools
and let them orchestrate.
"""

import json
import uuid
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

from config.settings import settings
from app.models.schemas import (
    AudioSegment, SegmentType, MusicDescription, Layer, Project,
    Instrument, Genre, RhythmPattern, MelodyContour
)
from app.services import rhythm_analyzer, pitch_analyzer, composium_bridge


# Initialize OpenAI client
client = AsyncOpenAI(api_key=settings.openai_api_key)


# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "assign_instrument",
            "description": "Assign an instrument to a musical segment. Use this to specify what instrument should play a melody or rhythm segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The ID of the musical segment"
                    },
                    "instrument": {
                        "type": "string",
                        "enum": ["piano", "guitar", "bass", "synth", "strings", "kick", "snare", "hi-hat", "clap", "cymbal"],
                        "description": "The instrument to assign"
                    }
                },
                "required": ["segment_id", "instrument"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_tempo",
            "description": "Set the project tempo in BPM",
            "parameters": {
                "type": "object",
                "properties": {
                    "bpm": {
                        "type": "integer",
                        "description": "Tempo in beats per minute (40-200)"
                    }
                },
                "required": ["bpm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_genre",
            "description": "Set the overall genre/style for the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "enum": ["hip-hop", "pop", "rock", "jazz", "electronic", "lo-fi", "rnb"],
                        "description": "The musical genre"
                    }
                },
                "required": ["genre"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "render_rhythm_layer",
            "description": "Render a rhythm segment as a percussion layer using MIDI synthesis",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The ID of the rhythm segment"
                    },
                    "instruments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of percussion instruments to use (kick, snare, hi-hat, clap, cymbal)"
                    }
                },
                "required": ["segment_id", "instruments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "render_melody_layer",
            "description": "Render a melody segment with a specified instrument sound",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The ID of the melody segment"
                    },
                    "instrument": {
                        "type": "string",
                        "enum": ["piano", "guitar", "bass", "synth", "strings"],
                        "description": "The melodic instrument to use"
                    }
                },
                "required": ["segment_id", "instrument"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "combine_layers",
            "description": "Mix multiple rendered layers together into the final output",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of layers to combine"
                    }
                },
                "required": ["layer_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete",
            "description": "Call this when you have finished processing all instructions and creating all layers",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of what was created"
                    }
                },
                "required": ["summary"]
            }
        }
    }
]


class OrchestratorState:
    """Holds state during orchestration."""

    def __init__(self, project: Project):
        self.project = project
        self.segment_data: dict[str, dict] = {}  # segment_id -> analyzed data
        self.instrument_assignments: dict[str, str] = {}  # segment_id -> instrument
        self.rendered_layers: list[Layer] = []
        self.genre: Optional[Genre] = None
        self.completed = False
        self.summary = ""


async def orchestrate(
    segments: list[AudioSegment],
    speech_instructions: list[str],
    segment_audio_data: dict[str, bytes],  # segment_id -> audio bytes
    project: Project,
) -> tuple[Project, str]:
    """
    Use OpenAI to orchestrate the music creation pipeline.

    Args:
        segments: List of classified audio segments
        speech_instructions: List of extracted speech transcripts (instructions)
        segment_audio_data: Dict mapping segment IDs to their audio bytes
        project: The project being built

    Returns:
        Tuple of (updated project, summary text for TTS feedback)
    """
    logger.info("=" * 60)
    logger.info("AGENT: Starting orchestration")
    logger.info(f"AGENT: Segments: {len(segments)}")
    logger.info(f"AGENT: Speech instructions: {speech_instructions}")
    logger.info(f"AGENT: Audio data for {len(segment_audio_data)} segments")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set! Add it to your .env file.")

    state = OrchestratorState(project)

    # Pre-analyze musical segments
    logger.info("AGENT: Pre-analyzing musical segments...")
    for segment in segments:
        if segment.type != SegmentType.SPEECH and segment.id in segment_audio_data:
            audio_bytes = segment_audio_data[segment.id]
            logger.info(f"AGENT: Analyzing segment {segment.id[:8]} ({segment.type.value}, {len(audio_bytes)} bytes)")

            if segment.type == SegmentType.RHYTHM:
                rhythm, onsets = await rhythm_analyzer.analyze_rhythm(audio_bytes)
                state.segment_data[segment.id] = {
                    "type": "rhythm",
                    "rhythm": rhythm,
                    "onsets": onsets
                }
                logger.info(f"AGENT:   -> Rhythm: {len(rhythm.beats)} beats, {rhythm.bpm} BPM")
            elif segment.type == SegmentType.MELODY:
                melody = await pitch_analyzer.analyze_melody(audio_bytes)
                state.segment_data[segment.id] = {
                    "type": "melody",
                    "melody": melody
                }
                logger.info(f"AGENT:   -> Melody: {len(melody.pitches)} pitches detected")

    # Build the context message for OpenAI
    context = _build_context_message(segments, speech_instructions, state.segment_data)
    logger.info("AGENT: Context message for AI:")
    logger.info("-" * 40)
    logger.info(context)
    logger.info("-" * 40)

    # Run the agent loop
    messages = [
        {
            "role": "system",
            "content": """You are a music production assistant that helps create tracks from user voice recordings.

You have been given:
1. Speech segments containing the user's instructions
2. Musical segments (melodies and rhythms) detected from their humming/beatboxing
3. Analysis data for each musical segment

Your job is to:
1. Interpret the user's instructions
2. Assign appropriate instruments to each musical segment
3. Set the tempo and genre based on the user's preferences
4. Render layers and combine them into a final track

You have access to a MIDI rendering engine (Composium) that produces professional-quality
instrument sounds. When you render a melody layer, the engine will automatically generate
appropriate accompaniment (chords, bass lines) for the chosen instrument. Available melodic
instruments: piano, guitar, bass, synth, strings.

Use the provided tools to accomplish this. When you're done, call the 'complete' tool with a summary.

Be creative but follow the user's explicit instructions. If they say "make this a piano melody", assign piano to that segment."""
        },
        {
            "role": "user",
            "content": context
        }
    ]

    max_iterations = 10
    logger.info(f"AGENT: Starting AI loop (max {max_iterations} iterations)")
    for iteration in range(max_iterations):
        if state.completed:
            logger.info(f"AGENT: Completed at iteration {iteration}")
            break

        logger.info(f"AGENT: Iteration {iteration + 1} - calling GPT-4o...")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.content:
            logger.info(f"AGENT: AI response text: {assistant_message.content}")

        if not assistant_message.tool_calls:
            logger.info("AGENT: No tool calls, ending loop")
            break

        # Process tool calls
        logger.info(f"AGENT: Processing {len(assistant_message.tool_calls)} tool calls")
        for tool_call in assistant_message.tool_calls:
            logger.info(f"AGENT: Tool call: {tool_call.function.name}({tool_call.function.arguments})")
            result = await _execute_tool(tool_call, state, segments, segment_audio_data)
            logger.info(f"AGENT: Tool result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    # Update project with rendered layers
    logger.info(f"AGENT: Loop finished. Rendered {len(state.rendered_layers)} layers")
    project.layers.extend(state.rendered_layers)
    if state.genre:
        if project.description is None:
            project.description = MusicDescription(genre=state.genre, instruments=[])
        else:
            project.description.genre = state.genre

    project.updated_at = datetime.now()

    logger.info(f"AGENT: Final summary: {state.summary or 'Track created successfully'}")
    logger.info(f"AGENT: Project has {len(project.layers)} total layers")
    logger.info("=" * 60)

    return project, state.summary or "Track created successfully"


def _resolve_segment_id(short_id: str, state: OrchestratorState) -> str | None:
    """Resolve a possibly-truncated segment ID to the full ID in state."""
    if short_id in state.segment_data:
        return short_id
    # Try prefix match (GPT-4o sometimes uses truncated IDs)
    for full_id in state.segment_data:
        if full_id.startswith(short_id):
            return full_id
    return None


async def _execute_tool(
    tool_call,
    state: OrchestratorState,
    segments: list[AudioSegment],
    segment_audio: dict[str, bytes],
) -> dict:
    """Execute a tool call and return the result."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    try:
        if name == "assign_instrument":
            segment_id = args["segment_id"]
            instrument = args["instrument"]
            state.instrument_assignments[segment_id] = instrument
            return {"success": True, "message": f"Assigned {instrument} to segment {segment_id[:8]}"}

        elif name == "set_tempo":
            bpm = args["bpm"]
            bpm = max(40, min(200, bpm))
            state.project.bpm = bpm
            return {"success": True, "message": f"Set tempo to {bpm} BPM"}

        elif name == "set_genre":
            genre_str = args["genre"]
            try:
                state.genre = Genre(genre_str)
                return {"success": True, "message": f"Set genre to {genre_str}"}
            except ValueError:
                return {"success": False, "error": f"Unknown genre: {genre_str}"}

        elif name == "render_rhythm_layer":
            segment_id = _resolve_segment_id(args["segment_id"], state)
            instruments = args.get("instruments", ["kick", "snare", "hi-hat"])

            if segment_id is None:
                return {"success": False, "error": "Segment not found or not analyzed"}

            data = state.segment_data[segment_id]
            if data["type"] != "rhythm":
                return {"success": False, "error": "Segment is not a rhythm segment"}

            rhythm: RhythmPattern = data["rhythm"]

            # Assign instruments to beats
            instrument_enums = [Instrument(i) for i in instruments if i in [e.value for e in Instrument]]
            rhythm = rhythm_analyzer.assign_instruments(rhythm, instrument_enums)

            audio_file = await composium_bridge.render_rhythm(
                rhythm, state.project.bpm, state.project.key_signature
            )

            layer = Layer(
                id=str(uuid.uuid4()),
                name=f"Rhythm - {', '.join(instruments)}",
                instrument=instrument_enums[0] if instrument_enums else None,
                segment_type=SegmentType.RHYTHM,
                rhythm=rhythm,
                audio_file=audio_file,
            )
            state.rendered_layers.append(layer)

            return {"success": True, "layer_id": layer.id, "message": f"Rendered rhythm layer with {instruments}"}

        elif name == "render_melody_layer":
            segment_id = _resolve_segment_id(args["segment_id"], state)
            instrument = args["instrument"]

            if segment_id is None:
                return {"success": False, "error": "Segment not found or not analyzed"}

            data = state.segment_data[segment_id]
            if data["type"] != "melody":
                return {"success": False, "error": "Segment is not a melody segment"}

            melody: MelodyContour = data["melody"]

            audio_file = await composium_bridge.render_melody(
                melody, instrument, state.project.bpm, state.project.key_signature
            )

            try:
                inst_enum = Instrument(instrument)
            except ValueError:
                inst_enum = Instrument.PIANO

            layer = Layer(
                id=str(uuid.uuid4()),
                name=f"Melody - {instrument}",
                instrument=inst_enum,
                segment_type=SegmentType.MELODY,
                melody=melody,
                audio_file=audio_file,
            )
            state.rendered_layers.append(layer)

            return {"success": True, "layer_id": layer.id, "message": f"Rendered melody layer with {instrument}"}

        elif name == "combine_layers":
            layer_ids = args.get("layer_ids", [])
            # For now, just acknowledge - actual mixing happens at project level
            return {"success": True, "message": f"Marked {len(layer_ids)} layers for mixing"}

        elif name == "complete":
            state.completed = True
            state.summary = args.get("summary", "Track created")
            return {"success": True, "message": "Orchestration complete"}

        else:
            return {"success": False, "error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def _build_context_message(
    segments: list[AudioSegment],
    speech_instructions: list[str],
    segment_data: dict[str, dict],
) -> str:
    """Build a context message describing the input for OpenAI."""
    lines = ["# User Recording Analysis\n"]

    lines.append("## Speech Instructions (in order):")
    for i, text in enumerate(speech_instructions, 1):
        lines.append(f"{i}. \"{text}\"")
    lines.append("")

    lines.append("## Musical Segments:")
    for segment in segments:
        if segment.type == SegmentType.SPEECH:
            continue

        data = segment_data.get(segment.id, {})
        duration = segment.end_seconds - segment.start_seconds

        lines.append(f"\n### Segment {segment.id} ({segment.type.value})")
        lines.append(f"- Duration: {duration:.2f}s")
        lines.append(f"- Time: {segment.start_seconds:.2f}s - {segment.end_seconds:.2f}s")

        if data.get("type") == "rhythm":
            rhythm = data.get("rhythm")
            if rhythm:
                lines.append(f"- Detected BPM: {rhythm.bpm}")
                lines.append(f"- Beats detected: {len(rhythm.beats)}")
                lines.append(f"- Bars: {rhythm.bars}")

        elif data.get("type") == "melody":
            melody = data.get("melody")
            if melody:
                lines.append(f"- Notes detected: {len(melody.pitches)}")
                if melody.key_signature:
                    lines.append(f"- Detected key: {melody.key_signature}")
                if melody.tempo_bpm:
                    lines.append(f"- Detected tempo: {melody.tempo_bpm} BPM")

    lines.append("\n## Your Task:")
    lines.append("Based on the user's instructions, assign instruments to the musical segments and render the layers.")
    lines.append("Use the tools provided to accomplish this.")

    return "\n".join(lines)
