"""High-level composition — combine multiple instruments into one Score."""

from __future__ import annotations

import math

from composium.notation import Analysis, Score, Voice
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


def _arrange_melody(analysis: Analysis) -> Score:
    """Create a melody voice from the raw analysis notes (GM Flute)."""
    tempo = analysis.tempo or 120
    beats_per_measure = 4.0
    spb = 60.0 / tempo

    total_beats = analysis.duration / spb
    target_measures = max(1, math.ceil(total_beats / beats_per_measure))
    score_duration = target_measures * beats_per_measure * spb

    voice = Voice(
        notes=list(analysis.notes),
        name="Melody",
        midi_program=73,  # GM Flute
        clef="treble",
    )
    return Score(
        voices=[voice],
        tempo=tempo,
        key=analysis.key,
        time_sig=(4, 4),
        duration=score_duration,
    )


def compose(analysis: Analysis, instruments: list[str]) -> Score:
    """Arrange multiple instruments and merge into a single Score.

    *instruments* is a list of names.  Valid names:
    ``"piano"``, ``"guitar"``, ``"drums"``, ``"melody"``
    (``"melody"`` renders the raw detected notes as a flute voice).
    """
    if not instruments:
        raise ValueError("instruments list must not be empty")

    all_voices: list[Voice] = []
    max_duration = 0.0

    for name in instruments:
        if name == "melody":
            score = _arrange_melody(analysis)
        elif name in _INSTRUMENTS:
            score = _INSTRUMENTS[name](analysis)
        else:
            valid = sorted(list(_INSTRUMENTS) + ["melody"])
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
