"""
Composium Bridge — adapts VibeBeat data structures to Composium's
MIDI rendering pipeline.

Replaces the old sine-wave melody renderer and sample-based rhythm
renderer with Composium's ABC -> abc2midi -> timidity -> ffmpeg chain.
"""

import asyncio
import uuid
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from composium.notation import Note, Voice, Score, Analysis
from composium.compose import compose
from composium.render import render

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import settings
from app.models.schemas import MelodyContour, RhythmPattern

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1a. Instrument mapping: VibeBeat instrument name -> Composium instrument list
# ---------------------------------------------------------------------------

_INSTRUMENT_MAP: dict[str, list[str]] = {
    "piano":   ["piano"],
    "guitar":  ["guitar"],
    "strings": ["piano"],       # closest GM match
    "synth":   ["edm"],         # EDM has synth bass + pads
    "bass":    ["piano"],       # piano includes Alberti bass voice
}

# ---------------------------------------------------------------------------
# 1b. Percussion mapping: VibeBeat instrument name -> GM MIDI note
# ---------------------------------------------------------------------------

_PERC_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "hi-hat": 42,
    "clap": 39,
    "cymbal": 49,
}


def _parse_key(key_signature: str | None) -> str:
    """Parse VibeBeat key string (e.g. 'C major', 'A minor') to Composium key (e.g. 'C', 'Am')."""
    if not key_signature:
        return "C"
    parts = key_signature.strip().split()
    if len(parts) >= 2 and parts[1].lower().startswith("min"):
        return parts[0] + "m"
    return parts[0]


# ---------------------------------------------------------------------------
# 1c. melody_to_analysis
# ---------------------------------------------------------------------------

def melody_to_analysis(
    melody: MelodyContour,
    bpm: int,
    key: str | None = None,
) -> Analysis:
    """Convert a VibeBeat MelodyContour to a Composium Analysis."""
    spb = 60.0 / bpm
    notes = [
        Note(
            midi_pitch=p.midi_note,
            start_beat=p.time_seconds / spb,
            duration_beats=p.duration_seconds / spb,
        )
        for p in melody.pitches
    ]

    parsed_key = _parse_key(key or melody.key_signature)

    if melody.pitches:
        last = melody.pitches[-1]
        duration = last.time_seconds + last.duration_seconds
    else:
        duration = 0.0

    return Analysis(
        tempo=bpm,
        key=parsed_key,
        duration=duration,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 1d. rhythm_to_score
# ---------------------------------------------------------------------------

def rhythm_to_score(
    rhythm: RhythmPattern,
    bpm: int,
    key: str | None = None,
) -> Score:
    """Convert a VibeBeat RhythmPattern to a Composium Score with a drum Voice on channel 10."""
    notes: list[Note] = []
    for beat in rhythm.beats:
        inst_name = beat.instrument.value
        midi_pitch = _PERC_MAP.get(inst_name)
        if midi_pitch is None:
            continue
        start_beat = beat.bar * 4 + beat.position * 0.25
        notes.append(Note(
            midi_pitch=midi_pitch,
            start_beat=start_beat,
            duration_beats=0.25,
        ))

    drum_voice = Voice(
        notes=notes,
        name="Drums",
        midi_program=0,
        clef="perc",
        midi_channel=10,
    )

    parsed_key = _parse_key(key)
    total_beats = rhythm.bars * 4
    duration = total_beats * (60.0 / bpm)

    return Score(
        voices=[drum_voice],
        tempo=bpm,
        key=parsed_key,
        time_sig=(4, 4),
        duration=duration,
    )


# ---------------------------------------------------------------------------
# 1e. render_melody  (replaces _render_melody_audio)
# ---------------------------------------------------------------------------

async def render_melody(
    melody: MelodyContour,
    instrument: str,
    bpm: int,
    key: str | None = None,
) -> str:
    """Render a melody via Composium's MIDI pipeline. Returns path to MP3."""
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"melody_{uuid.uuid4().hex[:8]}.mp3")

    analysis = melody_to_analysis(melody, bpm, key)
    instruments = _INSTRUMENT_MAP.get(instrument, ["piano"])

    logger.info(f"composium_bridge: rendering melody instrument={instrument} -> {instruments}")

    score = await asyncio.to_thread(compose, analysis, instruments)
    await asyncio.to_thread(render, score, output_path)
    return output_path


# ---------------------------------------------------------------------------
# 1f. render_rhythm  (replaces track_assembler.render_layer)
# ---------------------------------------------------------------------------

async def render_rhythm(
    rhythm: RhythmPattern,
    bpm: int,
    key: str | None = None,
) -> str:
    """Render a rhythm pattern via Composium's MIDI pipeline. Returns path to MP3."""
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"rhythm_{uuid.uuid4().hex[:8]}.mp3")

    score = rhythm_to_score(rhythm, bpm, key)

    logger.info(f"composium_bridge: rendering rhythm ({len(rhythm.beats)} beats, {bpm} BPM)")

    await asyncio.to_thread(render, score, output_path)
    return output_path
