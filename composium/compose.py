"""High-level composition — combine multiple instruments into one Score."""

from __future__ import annotations

import math

from composium.notation import Analysis, Note, Score, Voice
from composium.instruments.piano import arrange_piano
from composium.instruments.guitar import arrange_guitar
from composium.instruments.drums import arrange_drums
from composium.instruments.edm import arrange_edm

_INSTRUMENTS: dict[str, object] = {
    "piano": arrange_piano,
    "guitar": arrange_guitar,
    "drums": arrange_drums,
    "edm": arrange_edm,
}

# MIDI program numbers for simple melody rendering
_MIDI_PROGRAMS: dict[str, int] = {
    "piano": 0,       # Acoustic Grand Piano
    "guitar": 25,     # Acoustic Guitar (steel)
    "strings": 48,    # String Ensemble 1
    "synth": 81,      # Lead 1 (square)
    "bass": 33,       # Electric Bass (finger)
    "flute": 73,      # Flute
    "violin": 40,     # Violin
}


def _apply_legato(notes: list[Note], overlap: float = 0.1) -> list[Note]:
    """Extend note durations to create smooth, connected playback.

    Args:
        notes: List of Note objects
        overlap: Extra beats to extend each note (creates slight overlap)

    Returns:
        Notes with extended durations for legato effect
    """
    if not notes:
        return notes

    result = []
    sorted_notes = sorted(notes, key=lambda n: n.start_beat)

    for i, note in enumerate(sorted_notes):
        if i < len(sorted_notes) - 1:
            next_note = sorted_notes[i + 1]
            # Extend duration to reach the next note (with slight overlap)
            gap = next_note.start_beat - note.start_beat
            new_duration = max(note.duration_beats, gap + overlap)
        else:
            # Last note: extend slightly
            new_duration = note.duration_beats + overlap

        result.append(Note(
            midi_pitch=note.midi_pitch,
            start_beat=note.start_beat,
            duration_beats=new_duration,
        ))

    return result


def _arrange_melody(analysis: Analysis, midi_program: int = 73) -> Score:
    """Create a melody voice from the raw analysis notes.

    Args:
        analysis: The Analysis with detected notes
        midi_program: MIDI program number (instrument). Common values:
            0 = Acoustic Grand Piano
            25 = Acoustic Guitar (steel)
            40 = Violin
            73 = Flute (default)
    """
    tempo = analysis.tempo or 120
    beats_per_measure = 4.0
    spb = 60.0 / tempo

    total_beats = analysis.duration / spb
    target_measures = max(1, math.ceil(total_beats / beats_per_measure))
    score_duration = target_measures * beats_per_measure * spb

    # Apply legato: extend each note to connect with the next
    notes = _apply_legato(list(analysis.notes))

    voice = Voice(
        notes=notes,
        name="Melody",
        midi_program=midi_program,
        clef="treble",
    )
    return Score(
        voices=[voice],
        tempo=tempo,
        key=analysis.key,
        time_sig=(4, 4),
        duration=score_duration,
    )


def compose(analysis: Analysis, instruments: list[str], simple: bool = False) -> Score:
    """Arrange multiple instruments and merge into a single Score.

    Args:
        analysis: The Analysis with detected notes
        instruments: List of instrument names. Valid names:
            "piano", "guitar", "drums", "edm", "melody",
            "simple:piano", "simple:guitar", etc.
        simple: If True, render all instruments as simple melodies
            (just the detected notes, no accompaniment)

    Simple mode instruments render only the detected notes with the
    appropriate MIDI sound, no accompaniment or harmonies added.
    Use "simple:guitar" or set simple=True for accurate note playback.
    """
    if not instruments:
        raise ValueError("instruments list must not be empty")

    all_voices: list[Voice] = []
    max_duration = 0.0

    for name in instruments:
        # Handle simple melody mode: "simple:guitar" or "simple:piano"
        if name.startswith("simple:"):
            inst_name = name.split(":", 1)[1]
            midi_prog = _MIDI_PROGRAMS.get(inst_name, 0)
            score = _arrange_melody(analysis, midi_program=midi_prog)
        elif name == "melody":
            score = _arrange_melody(analysis)
        elif simple and name in _MIDI_PROGRAMS:
            # Simple mode: render as melody with correct instrument sound
            midi_prog = _MIDI_PROGRAMS.get(name, 0)
            score = _arrange_melody(analysis, midi_program=midi_prog)
        elif name in _INSTRUMENTS:
            score = _INSTRUMENTS[name](analysis)
        else:
            valid = sorted(list(_INSTRUMENTS) + ["melody"] + [f"simple:{k}" for k in _MIDI_PROGRAMS])
            raise ValueError(
                f"Unknown instrument: {name!r}. Choose from: {valid}"
            )

        all_voices.extend(score.voices)
        max_duration = max(max_duration, score.duration)

    return Score(
        voices=all_voices,
        tempo=analysis.tempo or 120,
        key=analysis.key,
        time_sig=(4, 4),
        duration=max_duration,
    )
