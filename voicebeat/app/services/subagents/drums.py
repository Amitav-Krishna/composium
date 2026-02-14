"""
Drums Subagent — specialised for percussion layer creation.

Tools: create_rhythm_layer, render_rhythm_layer, extend_rhythm_layer, done.
"""

import json
import uuid
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.models.schemas import (
    Layer, Project, Instrument, SegmentType,
    RhythmPattern, QuantizedBeat,
)
from app.services import rhythm_analyzer, composium_bridge
from .base import BaseSubagent, audio_duration, resolve_segment_id, build_beats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

DRUMS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_rhythm_layer",
            "description": (
                "Create a drum layer from scratch. You MUST provide beats for EVERY bar — "
                "bars without beats are SILENT. The rendered duration = bars * 4 * (60/BPM) seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Layer name, e.g. 'Verse Drums'",
                    },
                    "beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "integer", "description": "0-15 sixteenth-note position in bar (0=beat 1, 4=beat 2, 8=beat 3, 12=beat 4)"},
                                "bar": {"type": "integer", "description": "Bar number (0-indexed). Provide beats in EVERY bar."},
                                "instrument": {"type": "string", "enum": ["kick", "snare", "hi-hat", "clap", "cymbal"]},
                                "velocity": {"type": "number", "description": "0.0–1.0"},
                            },
                            "required": ["position", "bar", "instrument"],
                        },
                        "description": "Beat events covering ALL bars.",
                    },
                    "bars": {
                        "type": "integer",
                        "description": "Total bar count. Default 4.",
                    },
                },
                "required": ["name", "beats"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_rhythm_layer",
            "description": (
                "Render the user's beatboxing/tapping segment as a percussion layer. "
                "WARNING: user segments are typically short (1-2 bars)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "ID of the rhythm segment",
                    },
                    "instruments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Percussion instruments to use (kick, snare, hi-hat, clap, cymbal)",
                    },
                },
                "required": ["segment_id", "instruments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extend_rhythm_layer",
            "description": (
                "Add more bars and beats to an existing rhythm layer. "
                "You MUST provide additional_beats for the new bars — "
                "setting new_total_bars alone adds silent bars."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "string",
                        "description": "ID of the layer to extend",
                    },
                    "additional_beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "integer"},
                                "bar": {"type": "integer"},
                                "instrument": {"type": "string", "enum": ["kick", "snare", "hi-hat", "clap", "cymbal"]},
                                "velocity": {"type": "number"},
                            },
                            "required": ["position", "bar", "instrument"],
                        },
                        "description": "New beat events for the added bars.",
                    },
                    "new_total_bars": {
                        "type": "integer",
                        "description": "New total bar count (must be >= current).",
                    },
                },
                "required": ["layer_id", "additional_beats"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that all drum parts are complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# DrumsSubagent
# ---------------------------------------------------------------------------

class DrumsSubagent(BaseSubagent):
    name = "drums"
    max_iterations = 6

    def get_system_prompt(self, project: Project, instructions: dict) -> str:
        genre = instructions.get("genre", "pop")
        return f"""You are a drum programmer. You write percussion patterns for a {genre} track.

## Grid format
Each bar has 16 positions (sixteenth notes):
- 0 = beat 1, 4 = beat 2, 8 = beat 3, 12 = beat 4
- Positions 2,6,10,14 = eighth-note offsets
- Odd positions = sixteenth-note ghost notes

## Instruments
kick, snare, hi-hat, clap, cymbal

## Genre guidance
- hip-hop: heavy kick on 1, snare on 3, hi-hat on every eighth, occasional ghost notes
- pop: four-on-the-floor kick, snare on 2 & 4, open/closed hi-hat patterns
- rock: driving kick/snare, crash cymbal on transitions
- jazz: ride cymbal swing, kick comping, snare ghost notes
- electronic: punchy kick, layered claps, fast hi-hat rolls
- lo-fi: relaxed kick/snare, shuffled hi-hat, low velocities

## Rules
1. Provide beats for EVERY assigned bar — empty bars = silence
2. Keep patterns musical: vary velocity for groove (0.6–1.0)
3. For long sections, vary the pattern slightly every 2-4 bars
4. Call `done` when finished"""

    def get_tools(self) -> list[dict]:
        return DRUMS_TOOLS

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
        segment_audio = instructions.get("segment_audio", {})

        try:
            if name == "create_rhythm_layer":
                return await self._create(args, project, created_layers)
            elif name == "render_rhythm_layer":
                return await self._render(args, project, segment_data, segment_audio, created_layers)
            elif name == "extend_rhythm_layer":
                return await self._extend(args, project, created_layers)
            else:
                return {"success": False, "error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"DRUMS: Tool {name} failed: {e}")
            return {"success": False, "error": str(e)}

    # -- tool implementations -----------------------------------------------

    async def _create(self, args: dict, project: Project, created_layers: list[Layer]) -> dict:
        layer_name = args.get("name", "Drums")
        beats_data = args.get("beats", [])
        bars = args.get("bars", 4)

        beats = build_beats(beats_data)
        rhythm = RhythmPattern(beats=beats, bpm=project.bpm, bars=bars)

        audio_file = await composium_bridge.render_rhythm(
            rhythm, project.bpm, project.key_signature
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
        created_layers.append(layer)

        dur = audio_duration(audio_file)
        bars_with_beats = len(set(b.bar for b in beats))
        return {
            "success": True,
            "layer_id": layer.id,
            "audio_duration_sec": round(dur, 1),
            "bars": bars,
            "bars_with_beats": bars_with_beats,
            "total_beats": len(beats),
            "instruments": instruments_used,
            "message": f"Created '{layer_name}' — {dur:.1f}s, {bars} bars ({bars_with_beats} with content)",
        }

    async def _render(
        self, args: dict, project: Project,
        segment_data: dict, segment_audio: dict,
        created_layers: list[Layer],
    ) -> dict:
        segment_id = resolve_segment_id(args["segment_id"], segment_data)
        instruments = args.get("instruments", ["kick", "snare", "hi-hat"])

        if segment_id is None:
            return {"success": False, "error": "Segment not found or not analyzed"}

        data = segment_data[segment_id]
        if data["type"] != "rhythm":
            return {"success": False, "error": "Segment is not a rhythm segment"}

        rhythm: RhythmPattern = data["rhythm"]
        instrument_enums = [Instrument(i) for i in instruments if i in [e.value for e in Instrument]]
        rhythm = rhythm_analyzer.assign_instruments(rhythm, instrument_enums)

        audio_file = await composium_bridge.render_rhythm(
            rhythm, project.bpm, project.key_signature
        )

        layer = Layer(
            id=str(uuid.uuid4()),
            name=f"Rhythm - {', '.join(instruments)}",
            instrument=instrument_enums[0] if instrument_enums else None,
            segment_type=SegmentType.RHYTHM,
            rhythm=rhythm,
            audio_file=audio_file,
        )
        created_layers.append(layer)

        dur = audio_duration(audio_file)
        spb = 60.0 / project.bpm
        dur_bars = dur / (4 * spb) if spb > 0 else 0
        return {
            "success": True,
            "layer_id": layer.id,
            "audio_duration_sec": round(dur, 1),
            "bars": rhythm.bars,
            "message": f"Rendered rhythm — {dur:.1f}s, {dur_bars:.1f} bars",
        }

    async def _extend(self, args: dict, project: Project, created_layers: list[Layer]) -> dict:
        layer_id = args["layer_id"]
        additional_beats_data = args.get("additional_beats", [])
        new_total_bars = args.get("new_total_bars")

        target = None
        for lyr in created_layers:
            if lyr.id == layer_id or lyr.id.startswith(layer_id):
                target = lyr
                break

        if target is None:
            return {"success": False, "error": f"Layer '{layer_id}' not found"}

        if not target.rhythm:
            return {"success": False, "error": "Layer has no rhythm data"}

        for b in additional_beats_data:
            try:
                inst = Instrument(b.get("instrument", "kick"))
            except ValueError:
                inst = Instrument.KICK
            target.rhythm.beats.append(QuantizedBeat(
                position=b.get("position", 0),
                bar=b.get("bar", 0),
                instrument=inst,
                velocity=b.get("velocity", 1.0),
            ))

        if target.rhythm.beats:
            max_bar = max(b.bar for b in target.rhythm.beats)
            target.rhythm.bars = max(target.rhythm.bars, max_bar + 1, new_total_bars or 0)

        audio_file = await composium_bridge.render_rhythm(
            target.rhythm, project.bpm, project.key_signature
        )
        target.audio_file = audio_file

        dur = audio_duration(audio_file)
        bars_with_beats = len(set(b.bar for b in target.rhythm.beats))
        return {
            "success": True,
            "audio_duration_sec": round(dur, 1),
            "bars": target.rhythm.bars,
            "bars_with_beats": bars_with_beats,
            "message": f"Extended to {target.rhythm.bars} bars ({bars_with_beats} with content) — {dur:.1f}s",
        }
