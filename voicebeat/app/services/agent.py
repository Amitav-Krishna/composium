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
from app.services import rhythm_analyzer, pitch_analyzer, composium_bridge, vocal_processor


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
            "description": "Render a rhythm segment from analyzed audio as a percussion layer. WARNING: User audio segments are typically short (1-2 bars). The output layer will only be as long as the detected beats. If you need a longer layer, use create_rhythm_layer instead and write beats for all bars, or render first then extend_layer with additional_beats for the new bars.",
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
            "description": "Render a melody segment from analyzed audio with a specified instrument. WARNING: User audio segments are typically short (1-2 bars). The output layer will only cover the detected notes. If you need a longer layer, consider creating from scratch with create_melody_layer and writing notes for the full duration, or render first then extend_layer with additional_notes.",
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
            "description": "Call when done. This will REJECT if there are silent gaps (>2 bars) in the arrangement. If rejected, fix the gaps by adding/extending/moving layers, then call complete again.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "create_rhythm_layer",
            "description": "Create a new rhythm/drum layer from scratch. IMPORTANT: You must provide beats for EVERY bar. The 'bars' field only sets the total count — bars without beats will be SILENT. For example, if bars=4, you need beats with bar=0, bar=1, bar=2, and bar=3. The rendered audio duration will be: bars × 4 × (60/BPM) seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Layer name, e.g. 'Bass Drums'"
                    },
                    "beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "integer", "description": "0-15 sixteenth note position within bar (0=beat 1, 4=beat 2, 8=beat 3, 12=beat 4)"},
                                "bar": {"type": "integer", "description": "Bar number (0-indexed). You MUST provide beats in every bar from 0 to bars-1, otherwise those bars are silent."},
                                "instrument": {"type": "string", "enum": ["kick", "snare", "hi-hat", "clap", "cymbal"]},
                                "velocity": {"type": "number", "description": "0.0 to 1.0"}
                            },
                            "required": ["position", "bar", "instrument"]
                        },
                        "description": "Array of beat events. Must cover ALL bars — any bar without beats will be silent in the output audio."
                    },
                    "bars": {
                        "type": "integer",
                        "description": "Total number of bars. Only bars that have beats will produce sound. Default 2."
                    }
                },
                "required": ["name", "beats"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_melody_layer",
            "description": "Create a new melody layer from scratch. The AI provides the note sequence. Use when user asks to add a melodic instrument without providing audio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Layer name, e.g. 'Piano Melody'"
                    },
                    "instrument": {
                        "type": "string",
                        "enum": ["piano", "guitar", "bass", "synth", "strings"],
                        "description": "The melodic instrument to use"
                    },
                    "notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "midi_note": {"type": "integer", "description": "MIDI note number (e.g. 60=C4, 64=E4, 67=G4)"},
                                "start_beat": {"type": "number", "description": "Start position in beats from beginning"},
                                "duration_beats": {"type": "number", "description": "Duration in beats"}
                            },
                            "required": ["midi_note", "start_beat", "duration_beats"]
                        },
                        "description": "Array of note events"
                    }
                },
                "required": ["name", "instrument", "notes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_vocal_layer",
            "description": "Add the user's singing voice as a layer with autotune pitch correction. Use when the user wants their actual voice preserved (not converted to an instrument).",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The ID of the melody segment containing the singing"
                    },
                    "name": {
                        "type": "string",
                        "description": "Layer name, e.g. 'Vocals'"
                    }
                },
                "required": ["segment_id", "name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_layer",
            "description": "Remove an existing layer from the project by its ID or name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "string",
                        "description": "The ID or name of the layer to remove"
                    }
                },
                "required": ["layer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extend_layer",
            "description": "Extend an existing layer by adding more bars/notes. IMPORTANT: Setting new_total_bars alone does NOT add content — it only changes the bar count. You MUST also provide additional_beats (for rhythm) or additional_notes (for melody) with content for the new bars, otherwise the new bars will be silent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "string",
                        "description": "ID of the layer to extend"
                    },
                    "additional_beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "integer", "description": "0-15 sixteenth note position within bar"},
                                "bar": {"type": "integer", "description": "Bar number (0-indexed)"},
                                "instrument": {"type": "string", "enum": ["kick", "snare", "hi-hat", "clap", "cymbal"]},
                                "velocity": {"type": "number", "description": "0.0 to 1.0"}
                            },
                            "required": ["position", "bar", "instrument"]
                        },
                        "description": "Additional beat events for rhythm layers (same format as create_rhythm_layer)"
                    },
                    "additional_notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "midi_note": {"type": "integer", "description": "MIDI note number (e.g. 60=C4)"},
                                "start_beat": {"type": "number", "description": "Start position in beats from beginning"},
                                "duration_beats": {"type": "number", "description": "Duration in beats"}
                            },
                            "required": ["midi_note", "start_beat", "duration_beats"]
                        },
                        "description": "Additional note events for melody layers (same format as create_melody_layer)"
                    },
                    "new_total_bars": {
                        "type": "integer",
                        "description": "New total bar count (must be >= current bars)"
                    }
                },
                "required": ["layer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "arrange_layer",
            "description": "Move a layer to start at a specific bar position in the song. The layer plays from start_bar for its audio duration, then STOPS. It does NOT loop. To avoid silence gaps between sections, make sure layers are long enough to fill their section, or place the next layer immediately after the previous one ends. Check each layer's audio duration (shown in context) to plan arrangement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "string",
                        "description": "ID of the layer to reposition"
                    },
                    "start_bar": {
                        "type": "integer",
                        "description": "Bar number where this layer starts playing (0-indexed)"
                    }
                },
                "required": ["layer_id", "start_bar"]
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
            "content": """You are a music production assistant that creates tracks from user voice recordings.

## How the rendering engine works

Each layer produces an audio file with a specific duration. The duration is determined by content:
- Rhythm layers: duration = bars × 4 beats × (60/BPM) seconds. Only bars with beats produce sound.
- Melody layers: duration is determined by the notes provided. The engine adds accompaniment (chords, bass).

When you create a rhythm layer with bars=4, you MUST provide beats in bar=0, bar=1, bar=2, AND bar=3.
If you only put beats in bar=0, bars 1-3 will be SILENT — the 'bars' field just sets the total length.

## How arrangement works

arrange_layer sets where a layer starts playing. The layer plays for its audio duration, then STOPS.
It does NOT loop or repeat. Example at 120 BPM:
- A 4-bar drum layer (8.0s) at bar 0 → plays from 0s to 8.0s, then silence
- A 2-bar melody (4.0s) at bar 4 → plays from 8.0s to 12.0s, then silence
- Gap between bar 6 and anything at bar 8 = 4.0s of dead air

To avoid gaps: make layers long enough to cover their section, and place the next layer
where the previous one ends. Check audio durations in tool results.

## Your task

1. Interpret the user's instructions
2. Set tempo and genre
3. Render or create layers (providing beats/notes for ALL bars)
4. Arrange layers to create the requested song structure
5. Call 'complete' when done

Be creative but follow the user's explicit instructions."""
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


def _audio_duration(path: str) -> float:
    """Return audio file duration in seconds, or 0.0 on error."""
    try:
        from pydub import AudioSegment as PydubSegment
        return len(PydubSegment.from_file(path)) / 1000.0
    except Exception:
        return 0.0


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

            dur = _audio_duration(audio_file)
            spb = 60.0 / state.project.bpm
            dur_bars = dur / (4 * spb) if spb > 0 else 0
            warn = f" ⚠ Layer is only {dur_bars:.1f} bars — extend it or create a longer one from scratch to avoid gaps." if dur_bars < 2 else ""
            return {"success": True, "layer_id": layer.id, "audio_duration_sec": round(dur, 1), "bars": rhythm.bars, "beats_with_content": len(rhythm.beats), "message": f"Rendered rhythm layer with {instruments} — {dur:.1f}s audio, {dur_bars:.1f} bars{warn}"}

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

            dur = _audio_duration(audio_file)
            spb = 60.0 / state.project.bpm
            dur_bars = dur / (4 * spb) if spb > 0 else 0
            warn = f" ⚠ Layer is only {dur_bars:.1f} bars — extend it or create a longer one from scratch to avoid gaps." if dur_bars < 2 else ""
            return {"success": True, "layer_id": layer.id, "audio_duration_sec": round(dur, 1), "message": f"Rendered melody layer with {instrument} — {dur:.1f}s audio, {dur_bars:.1f} bars{warn}"}

        elif name == "combine_layers":
            layer_ids = args.get("layer_ids", [])
            # For now, just acknowledge - actual mixing happens at project level
            return {"success": True, "message": f"Marked {len(layer_ids)} layers for mixing"}

        elif name == "create_rhythm_layer":
            layer_name = args.get("name", "Rhythm")
            beats_data = args.get("beats", [])
            bars = args.get("bars", 2)

            from app.models.schemas import QuantizedBeat
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

            rhythm = RhythmPattern(beats=beats, bpm=state.project.bpm, bars=bars)

            audio_file = await composium_bridge.render_rhythm(
                rhythm, state.project.bpm, state.project.key_signature
            )

            instruments_used = list(set(b.instrument.value for b in beats))
            layer = Layer(
                id=str(uuid.uuid4()),
                name=layer_name,
                instrument=beats[0].instrument if beats else Instrument.KICK,
                segment_type=SegmentType.RHYTHM,
                rhythm=rhythm,
                audio_file=audio_file,
            )
            state.rendered_layers.append(layer)

            dur = _audio_duration(audio_file)
            bars_with_beats = len(set(b.bar for b in beats))
            return {"success": True, "layer_id": layer.id, "audio_duration_sec": round(dur, 1), "bars": bars, "bars_with_beats": bars_with_beats, "total_beats": len(beats), "message": f"Created rhythm layer '{layer_name}' — {dur:.1f}s audio, {bars} bars ({bars_with_beats} with content)"}

        elif name == "create_melody_layer":
            layer_name = args.get("name", "Melody")
            instrument = args.get("instrument", "piano")
            notes_data = args.get("notes", [])

            from app.models.schemas import PitchEvent
            spb = 60.0 / state.project.bpm
            pitches = []
            for n in notes_data:
                midi = n["midi_note"]
                start_beat = n["start_beat"]
                dur_beats = n["duration_beats"]
                freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
                note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                octave = (midi // 12) - 1
                note_name = f"{note_names[midi % 12]}{octave}"
                pitches.append(PitchEvent(
                    time_seconds=start_beat * spb,
                    frequency_hz=freq,
                    midi_note=midi,
                    note_name=note_name,
                    duration_seconds=dur_beats * spb,
                ))

            melody = MelodyContour(
                pitches=pitches,
                key_signature=state.project.key_signature,
                tempo_bpm=state.project.bpm,
            )

            audio_file = await composium_bridge.render_melody(
                melody, instrument, state.project.bpm, state.project.key_signature
            )

            try:
                inst_enum = Instrument(instrument)
            except ValueError:
                inst_enum = Instrument.PIANO

            layer = Layer(
                id=str(uuid.uuid4()),
                name=layer_name,
                instrument=inst_enum,
                segment_type=SegmentType.MELODY,
                melody=melody,
                audio_file=audio_file,
            )
            state.rendered_layers.append(layer)

            dur = _audio_duration(audio_file)
            return {"success": True, "layer_id": layer.id, "audio_duration_sec": round(dur, 1), "message": f"Created melody layer '{layer_name}' with {instrument} — {dur:.1f}s audio"}

        elif name == "add_vocal_layer":
            segment_id = args["segment_id"]
            layer_name = args.get("name", "Vocals")

            # Resolve possibly-truncated segment ID
            resolved_id = None
            for sid in segment_audio:
                if sid == segment_id or sid.startswith(segment_id):
                    resolved_id = sid
                    break

            if resolved_id is None:
                return {"success": False, "error": f"Segment {segment_id} not found"}

            audio_bytes = segment_audio[resolved_id]
            audio_file = await vocal_processor.autotune(
                audio_bytes,
                key_signature=state.project.key_signature,
            )

            layer = Layer(
                id=str(uuid.uuid4()),
                name=layer_name,
                instrument=Instrument.VOCAL,
                segment_type=SegmentType.VOCAL,
                audio_file=audio_file,
            )
            state.rendered_layers.append(layer)

            return {"success": True, "layer_id": layer.id, "message": f"Added autotuned vocal layer '{layer_name}'"}

        elif name == "extend_layer":
            layer_id = args["layer_id"]
            additional_beats = args.get("additional_beats", [])
            additional_notes = args.get("additional_notes", [])
            new_total_bars = args.get("new_total_bars")

            logger.info(f"EXTEND: layer_id={layer_id}, +beats={len(additional_beats)}, +notes={len(additional_notes)}, new_total_bars={new_total_bars}")

            # Find the layer by ID or prefix match
            target_layer = None
            for layer in state.project.layers:
                if layer.id == layer_id or layer.id.startswith(layer_id):
                    target_layer = layer
                    break
            # Also check rendered_layers (newly created in this session)
            if target_layer is None:
                for layer in state.rendered_layers:
                    if layer.id == layer_id or layer.id.startswith(layer_id):
                        target_layer = layer
                        break

            if target_layer is None:
                return {"success": False, "error": f"Layer '{layer_id}' not found"}

            logger.info(f"EXTEND: Found layer '{target_layer.name}' type={target_layer.segment_type.value}")

            from app.models.schemas import QuantizedBeat, PitchEvent

            if target_layer.segment_type == SegmentType.RHYTHM and target_layer.rhythm:
                old_bars = target_layer.rhythm.bars
                old_beat_count = len(target_layer.rhythm.beats)

                for b in additional_beats:
                    try:
                        inst = Instrument(b.get("instrument", "kick"))
                    except ValueError:
                        inst = Instrument.KICK
                    target_layer.rhythm.beats.append(QuantizedBeat(
                        position=b.get("position", 0),
                        bar=b.get("bar", 0),
                        instrument=inst,
                        velocity=b.get("velocity", 1.0),
                    ))
                # Auto-compute bar count from actual beat data
                if target_layer.rhythm.beats:
                    max_bar = max(b.bar for b in target_layer.rhythm.beats)
                    required_bars = max_bar + 1
                    target_layer.rhythm.bars = max(
                        target_layer.rhythm.bars,
                        required_bars,
                        new_total_bars or 0,
                    )

                logger.info(f"EXTEND: bars {old_bars} -> {target_layer.rhythm.bars}, beats {old_beat_count} -> {len(target_layer.rhythm.beats)}")

                audio_file = await composium_bridge.render_rhythm(
                    target_layer.rhythm, state.project.bpm, state.project.key_signature
                )
                target_layer.audio_file = audio_file
                dur = _audio_duration(audio_file)
                bars_with_beats = len(set(b.bar for b in target_layer.rhythm.beats))
                logger.info(f"EXTEND: Rendered to {audio_file} ({dur:.1f}s)")
                return {"success": True, "audio_duration_sec": round(dur, 1), "bars": target_layer.rhythm.bars, "bars_with_beats": bars_with_beats, "message": f"Extended rhythm to {target_layer.rhythm.bars} bars ({bars_with_beats} with content) — {dur:.1f}s audio"}

            elif target_layer.segment_type == SegmentType.MELODY and target_layer.melody:
                old_note_count = len(target_layer.melody.pitches)
                spb = 60.0 / state.project.bpm
                for n in additional_notes:
                    midi = n["midi_note"]
                    start_beat = n["start_beat"]
                    dur_beats = n["duration_beats"]
                    freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
                    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                    octave = (midi // 12) - 1
                    note_name = f"{note_names[midi % 12]}{octave}"
                    target_layer.melody.pitches.append(PitchEvent(
                        time_seconds=start_beat * spb,
                        frequency_hz=freq,
                        midi_note=midi,
                        note_name=note_name,
                        duration_seconds=dur_beats * spb,
                    ))
                logger.info(f"EXTEND: melody notes {old_note_count} -> {len(target_layer.melody.pitches)}")
                inst_str = target_layer.instrument.value if target_layer.instrument else "piano"
                audio_file = await composium_bridge.render_melody(
                    target_layer.melody, inst_str, state.project.bpm, state.project.key_signature
                )
                target_layer.audio_file = audio_file
                dur = _audio_duration(audio_file)
                logger.info(f"EXTEND: Rendered melody to {audio_file} ({dur:.1f}s)")
                return {"success": True, "audio_duration_sec": round(dur, 1), "message": f"Extended melody to {len(target_layer.melody.pitches)} notes — {dur:.1f}s audio"}

            else:
                return {"success": False, "error": "Layer has no rhythm or melody data to extend"}

        elif name == "arrange_layer":
            layer_id = args["layer_id"]
            start_bar = args["start_bar"]

            # Find the layer by ID or prefix match
            target_layer = None
            for layer in state.project.layers:
                if layer.id == layer_id or layer.id.startswith(layer_id):
                    target_layer = layer
                    break
            if target_layer is None:
                for layer in state.rendered_layers:
                    if layer.id == layer_id or layer.id.startswith(layer_id):
                        target_layer = layer
                        break

            if target_layer is None:
                return {"success": False, "error": f"Layer '{layer_id}' not found"}

            target_layer.start_bar = max(0, start_bar)
            dur = _audio_duration(target_layer.audio_file) if target_layer.audio_file else 0.0
            spb = 60.0 / state.project.bpm
            start_sec = target_layer.start_bar * 4 * spb
            end_sec = start_sec + dur
            end_bar = target_layer.start_bar + dur / (4 * spb) if spb > 0 else 0
            return {"success": True, "message": f"Moved '{target_layer.name}' to bar {target_layer.start_bar} (plays {start_sec:.1f}s–{end_sec:.1f}s, ends at bar {end_bar:.1f})"}

        elif name == "remove_layer":
            layer_id = args["layer_id"]
            # Try matching by ID or name
            for layer in list(state.project.layers):
                if layer.id == layer_id or layer.id.startswith(layer_id) or layer.name.lower() == layer_id.lower():
                    state.project.layers.remove(layer)
                    return {"success": True, "message": f"Removed layer '{layer.name}' ({layer.id[:8]})"}

            return {"success": False, "error": f"Layer '{layer_id}' not found"}

        elif name == "complete":
            # Check for gaps before allowing completion
            all_layers = list(state.project.layers) + list(state.rendered_layers)
            if all_layers and state.project.bpm:
                spb = 60.0 / state.project.bpm
                bar_sec = 4 * spb
                # Build coverage map: which seconds have audio
                intervals = []
                for lyr in all_layers:
                    if lyr.audio_file:
                        dur = _audio_duration(lyr.audio_file)
                        start = lyr.start_bar * bar_sec
                        intervals.append((start, start + dur))
                if intervals:
                    intervals.sort()
                    song_end = max(e for _, e in intervals)
                    # Find gaps > 2 bars
                    covered_until = 0.0
                    gaps = []
                    for start, end in intervals:
                        if start > covered_until + bar_sec * 2:
                            gaps.append((covered_until, start))
                        covered_until = max(covered_until, end)
                    if gaps:
                        gap_desc = "; ".join(
                            f"{g[0]:.1f}s–{g[1]:.1f}s (bar {g[0]/bar_sec:.0f}–{g[1]/bar_sec:.0f})"
                            for g in gaps
                        )
                        return {
                            "success": False,
                            "error": f"Cannot complete: the arrangement has silent gaps: {gap_desc}. "
                                     f"Add or extend layers to fill these gaps, or move existing layers to cover them. "
                                     f"Then call complete again."
                        }

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
    lines.append("Use arrange_layer to position layers at different bar offsets if the user wants song structure (e.g., intro, verse, chorus).")
    lines.append("Use the tools provided to accomplish this.")

    return "\n".join(lines)


async def orchestrate_refinement(
    project: Project,
    instructions: str,
    segments: list[AudioSegment] | None = None,
    segment_audio_data: dict[str, bytes] | None = None,
) -> tuple[Project, str]:
    """
    Refine an existing project based on user instructions.

    Can add new layers (from scratch or from audio), remove existing layers,
    or add vocal layers with autotune.

    Args:
        project: The existing project to refine
        instructions: Text instructions from the user
        segments: Optional new audio segments to incorporate
        segment_audio_data: Optional audio data for new segments

    Returns:
        Tuple of (updated project, summary text)
    """
    logger.info("=" * 60)
    logger.info("AGENT: Starting refinement orchestration")
    logger.info(f"AGENT: Instructions: {instructions}")
    logger.info(f"AGENT: Existing layers: {len(project.layers)}")
    logger.info(f"AGENT: New segments: {len(segments) if segments else 0}")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set! Add it to your .env file.")

    state = OrchestratorState(project)
    segment_audio_data = segment_audio_data or {}

    # Pre-analyze new musical segments (if any)
    if segments:
        for segment in segments:
            if segment.type not in (SegmentType.SPEECH,) and segment.id in segment_audio_data:
                audio_bytes = segment_audio_data[segment.id]
                logger.info(f"AGENT: Analyzing new segment {segment.id[:8]} ({segment.type.value})")

                if segment.type == SegmentType.RHYTHM:
                    rhythm, onsets = await rhythm_analyzer.analyze_rhythm(audio_bytes)
                    state.segment_data[segment.id] = {
                        "type": "rhythm",
                        "rhythm": rhythm,
                        "onsets": onsets,
                    }
                elif segment.type == SegmentType.MELODY:
                    melody = await pitch_analyzer.analyze_melody(audio_bytes)
                    state.segment_data[segment.id] = {
                        "type": "melody",
                        "melody": melody,
                    }

    # Build refinement context
    context = _build_refinement_context(project, instructions, segments, state.segment_data)
    logger.info("AGENT: Refinement context:")
    logger.info(context)

    messages = [
        {
            "role": "system",
            "content": """You are a music production assistant helping refine an existing track.

## How layers work (read carefully)

Each layer has an audio file with a specific duration (shown as "Xs audio" in the layer list).
- A layer at bar N plays from that bar's time offset for its audio duration, then STOPS. It does NOT loop.
- Rhythm audio duration = bars × 4 × (60/BPM) seconds. Only bars with beats produce sound.
- If you create a rhythm with bars=4 but only put beats in bar=0, you get 1 bar of drums and 3 bars of SILENCE.
- You MUST provide beats for EVERY bar (bar=0, bar=1, bar=2, bar=3 for a 4-bar layer).
- When extending a rhythm with extend_layer, you MUST provide additional_beats for the new bars — setting new_total_bars alone just adds silent bars.

## Arrangement and gaps

Layers play from their start_bar for their duration, then stop. Gaps between layers = silence in the output.
Use the audio duration shown for each layer to calculate where it ends:
  end_time = start_bar_offset + audio_duration
Place the next layer where the previous one ends to avoid gaps. Or overlap layers for density.

## What you can do

1. create_rhythm_layer / create_melody_layer — create from scratch (provide beats/notes for ALL bars)
2. remove_layer — remove existing layers
3. extend_layer — add content to new bars (must provide beats/notes, not just bar count)
4. render_rhythm_layer / render_melody_layer — render from user's audio
5. add_vocal_layer — add autotuned vocals
6. arrange_layer — position layers (check durations to avoid gaps)

When done, call 'complete' with a summary of what changed."""
        },
        {
            "role": "user",
            "content": context
        }
    ]

    max_iterations = 10
    for iteration in range(max_iterations):
        if state.completed:
            break

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.content:
            logger.info(f"AGENT: AI response: {assistant_message.content}")

        if not assistant_message.tool_calls:
            break

        for tool_call in assistant_message.tool_calls:
            logger.info(f"AGENT: Tool call: {tool_call.function.name}({tool_call.function.arguments})")
            result = await _execute_tool(tool_call, state, segments or [], segment_audio_data)
            logger.info(f"AGENT: Tool result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    # Add new layers to project
    project.layers.extend(state.rendered_layers)
    if state.genre:
        if project.description is None:
            project.description = MusicDescription(genre=state.genre, instruments=[])
        else:
            project.description.genre = state.genre

    project.updated_at = datetime.now()

    summary = state.summary or "Track refined successfully"
    logger.info(f"AGENT: Refinement summary: {summary}")
    logger.info(f"AGENT: Project now has {len(project.layers)} layers")
    logger.info("=" * 60)

    return project, summary


def _build_refinement_context(
    project: Project,
    instructions: str,
    segments: list[AudioSegment] | None,
    segment_data: dict[str, dict],
) -> str:
    """Build context message for refinement orchestration."""
    lines = ["# Current Project State\n"]

    # Project metadata
    genre_str = project.description.genre.value if project.description and project.description.genre else "not set"
    key_str = project.key_signature or "not set"
    lines.append(f"- BPM: {project.bpm}, Key: {key_str}, Genre: {genre_str}")
    lines.append("")

    # Existing layers
    lines.append("## Existing Layers:")
    if project.layers:
        for i, layer in enumerate(project.layers, 1):
            inst = layer.instrument.value if layer.instrument else layer.segment_type.value
            audio_dur = ""
            if layer.audio_file:
                try:
                    from pydub import AudioSegment as PydubSegment
                    a = PydubSegment.from_file(layer.audio_file)
                    audio_dur = f", {len(a)/1000:.1f}s audio"
                except Exception:
                    pass
            lines.append(f"{i}. [id:{layer.id}] {layer.name} - {inst} (bar {layer.start_bar}{audio_dur})")
    else:
        lines.append("(no layers yet)")
    lines.append("")

    # User's request
    lines.append("## User's Refinement Request:")
    lines.append(f'"{instructions}"')
    lines.append("")

    # New audio segments (if any)
    if segments:
        lines.append("## New Audio Segments:")
        for segment in segments:
            if segment.type == SegmentType.SPEECH:
                continue
            data = segment_data.get(segment.id, {})
            duration = segment.end_seconds - segment.start_seconds
            lines.append(f"\n### Segment {segment.id} ({segment.type.value})")
            lines.append(f"- Duration: {duration:.2f}s")

            if data.get("type") == "rhythm":
                rhythm = data.get("rhythm")
                if rhythm:
                    lines.append(f"- Detected BPM: {rhythm.bpm}")
                    lines.append(f"- Beats: {len(rhythm.beats)}")
            elif data.get("type") == "melody":
                melody = data.get("melody")
                if melody:
                    lines.append(f"- Notes: {len(melody.pitches)}")
                    if melody.key_signature:
                        lines.append(f"- Key: {melody.key_signature}")
        lines.append("")

    lines.append("## Your Task:")
    lines.append("Modify the project based on the user's request. You can add new layers, remove existing ones, or generate patterns from scratch.")
    lines.append("When done, call 'complete' with a summary of what changed.")

    return "\n".join(lines)
