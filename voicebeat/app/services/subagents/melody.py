"""
Melody Subagent — parameterised by instrument and role.

One class handles piano, guitar, bass, synth, and strings by receiving
the instrument and role (melody / chords / bass_line / pad) at init time.

Tools: create_melody_layer, render_melody_layer, extend_melody_layer, done.
"""

import json
import uuid
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.models.schemas import (
    Layer, Project, Instrument, SegmentType,
    MelodyContour, PitchEvent,
)
from app.services import composium_bridge
from .base import BaseSubagent, audio_duration, resolve_segment_id, build_pitches

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument ranges (MIDI note numbers)
# ---------------------------------------------------------------------------

_RANGES: dict[str, tuple[int, int]] = {
    "piano":   (36, 84),
    "guitar":  (40, 76),
    "bass":    (28, 55),
    "synth":   (36, 84),
    "strings": (36, 84),
}

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

MELODY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_melody_layer",
            "description": (
                "Create a melodic layer from scratch. Provide an array of notes "
                "(midi_note, start_beat, duration_beats). The engine also adds "
                "accompaniment (chords, bass) automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Layer name, e.g. 'Verse Piano'",
                    },
                    "notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "midi_note": {"type": "integer", "description": "MIDI note (e.g. 60=C4, 64=E4, 67=G4)"},
                                "start_beat": {"type": "number", "description": "Start position in beats from beginning"},
                                "duration_beats": {"type": "number", "description": "Duration in beats"},
                            },
                            "required": ["midi_note", "start_beat", "duration_beats"],
                        },
                        "description": "Array of note events",
                    },
                },
                "required": ["name", "notes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_melody_layer",
            "description": (
                "Render the user's hummed melody segment with the assigned instrument. "
                "WARNING: user segments are typically short (1-2 bars)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "ID of the melody segment",
                    },
                },
                "required": ["segment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extend_melody_layer",
            "description": (
                "Add more notes to an existing melody layer. "
                "Provide additional_notes for the new content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "string",
                        "description": "ID of the layer to extend",
                    },
                    "additional_notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "midi_note": {"type": "integer"},
                                "start_beat": {"type": "number"},
                                "duration_beats": {"type": "number"},
                            },
                            "required": ["midi_note", "start_beat", "duration_beats"],
                        },
                        "description": "New note events to add",
                    },
                },
                "required": ["layer_id", "additional_notes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that all melodic parts are complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# MelodySubagent
# ---------------------------------------------------------------------------

class MelodySubagent(BaseSubagent):
    name = "melody"
    max_iterations = 6

    def __init__(self, instrument: str = "piano", role: str = "melody"):
        self.instrument = instrument
        self.role = role
        self.name = f"{instrument}_{role}"

    def get_system_prompt(self, project: Project, instructions: dict) -> str:
        genre = instructions.get("genre", "pop")
        low, high = _RANGES.get(self.instrument, (36, 84))
        key = project.key_signature or "C major"

        role_guidance = {
            "melody": (
                "Write a singable main melody. Use stepwise motion with occasional leaps. "
                "Keep phrases 2-4 bars with clear contour (rise, peak, fall)."
            ),
            "chords": (
                "Write chord voicings. Use 3-4 note voicings spaced across beats. "
                "Common progressions: I-V-vi-IV, ii-V-I, I-vi-IV-V. "
                "Sustain chords for 2-4 beats each."
            ),
            "bass_line": (
                "Write a bass line following the chord roots. "
                "Use root notes on beat 1, passing tones between chords. "
                "Keep it simple and rhythmically locked to the kick drum."
            ),
            "pad": (
                "Write sustained pad chords for atmosphere. "
                "Use wide voicings (spread across 1-2 octaves), long durations (4-8 beats). "
                "Gentle velocity for ambient texture."
            ),
        }

        return f"""You are a {self.instrument} player writing {self.role} parts for a {genre} track in {key}.

## Note format
Each note: (midi_note, start_beat, duration_beats)
- midi_note: MIDI number (60=C4). Your range: {low}–{high}.
- start_beat: position in beats from the start (beat 0 = bar 0 beat 1)
- duration_beats: how long the note sounds

## {self.role.replace('_', ' ').title()} guidance
{role_guidance.get(self.role, role_guidance['melody'])}

## Rules
1. Stay within your range ({low}–{high})
2. Cover ALL assigned bars — calculate: bar N starts at beat N*4
3. Write musically: use the key signature, resolve tensions
4. Call `done` when finished"""

    def get_tools(self) -> list[dict]:
        return MELODY_TOOLS

    async def execute_tool(
        self,
        tool_call,
        project: Project,
        instructions: dict,
        created_layers: list[Layer],
    ) -> dict:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        segment_data = instructions.get("segment_data", {})

        try:
            if name == "create_melody_layer":
                return await self._create(args, project, created_layers)
            elif name == "render_melody_layer":
                return await self._render(args, project, segment_data, created_layers)
            elif name == "extend_melody_layer":
                return await self._extend(args, project, created_layers)
            else:
                return {"success": False, "error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"{self.name.upper()}: Tool {name} failed: {e}")
            return {"success": False, "error": str(e)}

    # -- tool implementations -----------------------------------------------

    async def _create(self, args: dict, project: Project, created_layers: list[Layer]) -> dict:
        layer_name = args.get("name", f"{self.instrument.title()} {self.role.title()}")
        notes_data = args.get("notes", [])

        pitches = build_pitches(notes_data, project.bpm)
        melody = MelodyContour(
            pitches=pitches,
            key_signature=project.key_signature,
            tempo_bpm=project.bpm,
        )

        audio_file = await composium_bridge.render_melody(
            melody, self.instrument, project.bpm, project.key_signature
        )

        try:
            inst_enum = Instrument(self.instrument)
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
        created_layers.append(layer)

        dur = audio_duration(audio_file)
        return {
            "success": True,
            "layer_id": layer.id,
            "audio_duration_sec": round(dur, 1),
            "message": f"Created '{layer_name}' with {self.instrument} — {dur:.1f}s, {len(pitches)} notes",
        }

    async def _render(
        self, args: dict, project: Project,
        segment_data: dict, created_layers: list[Layer],
    ) -> dict:
        segment_id = resolve_segment_id(args["segment_id"], segment_data)
        if segment_id is None:
            return {"success": False, "error": "Segment not found or not analyzed"}

        data = segment_data[segment_id]
        if data["type"] != "melody":
            return {"success": False, "error": "Segment is not a melody segment"}

        melody: MelodyContour = data["melody"]
        audio_file = await composium_bridge.render_melody(
            melody, self.instrument, project.bpm, project.key_signature
        )

        try:
            inst_enum = Instrument(self.instrument)
        except ValueError:
            inst_enum = Instrument.PIANO

        layer = Layer(
            id=str(uuid.uuid4()),
            name=f"Melody - {self.instrument}",
            instrument=inst_enum,
            segment_type=SegmentType.MELODY,
            melody=melody,
            audio_file=audio_file,
        )
        created_layers.append(layer)

        dur = audio_duration(audio_file)
        return {
            "success": True,
            "layer_id": layer.id,
            "audio_duration_sec": round(dur, 1),
            "message": f"Rendered melody with {self.instrument} — {dur:.1f}s",
        }

    async def _extend(self, args: dict, project: Project, created_layers: list[Layer]) -> dict:
        layer_id = args["layer_id"]
        additional_notes = args.get("additional_notes", [])

        target = None
        for lyr in created_layers:
            if lyr.id == layer_id or lyr.id.startswith(layer_id):
                target = lyr
                break

        if target is None:
            return {"success": False, "error": f"Layer '{layer_id}' not found"}

        if not target.melody:
            return {"success": False, "error": "Layer has no melody data"}

        spb = 60.0 / project.bpm
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for n in additional_notes:
            midi = n["midi_note"]
            start_beat = n["start_beat"]
            dur_beats = n["duration_beats"]
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            octave = (midi // 12) - 1
            note_name = f"{note_names[midi % 12]}{octave}"
            target.melody.pitches.append(PitchEvent(
                time_seconds=start_beat * spb,
                frequency_hz=freq,
                midi_note=midi,
                note_name=note_name,
                duration_seconds=dur_beats * spb,
            ))

        audio_file = await composium_bridge.render_melody(
            target.melody, self.instrument, project.bpm, project.key_signature
        )
        target.audio_file = audio_file

        dur = audio_duration(audio_file)
        return {
            "success": True,
            "audio_duration_sec": round(dur, 1),
            "message": f"Extended to {len(target.melody.pitches)} notes — {dur:.1f}s",
        }
