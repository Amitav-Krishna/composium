"""Drum-specific arrangement patterns."""

from __future__ import annotations

import math

from composium.notation import Analysis, Note, Score, Voice

# GM percussion map (channel 10)
KICK = 36  # Bass Drum 1
SNARE = 38  # Acoustic Snare
HIHAT_CLOSED = 42
HIHAT_OPEN = 46
CRASH = 49


def _rock_pattern(measures: int, beats_per_measure: float = 4.0) -> list[Note]:
    """Generate a standard rock beat for the given number of measures.

    Pattern per measure (eighth-note grid):
        Beat:  1   1+  2   2+  3   3+  4   4+
        Kick:  X           X
        Snare:     X           X
        HH:    X   X   X   X   X   X   X   X
    """
    notes: list[Note] = []
    grid = 0.5  # eighth notes

    for m in range(measures):
        offset = m * beats_per_measure
        slots = int(beats_per_measure / grid)  # 8 slots per 4/4 measure

        for slot in range(slots):
            beat = offset + slot * grid

            # Hi-hat on every eighth note
            notes.append(Note(HIHAT_CLOSED, beat, grid))

            # Kick on beats 1 and 3 (slots 0 and 4)
            if slot == 0 or slot == 4:
                notes.append(Note(KICK, beat, grid))

            # Snare on beats 2 and 4 (slots 2 and 6)
            if slot == 2 or slot == 6:
                notes.append(Note(SNARE, beat, grid))

    return notes


def arrange_drums(analysis: Analysis) -> Score:
    """Generate a single-voice drum Score from an Analysis."""
    tempo = analysis.tempo or 120
    beats_per_measure = 4.0  # 4/4 time
    spb = 60.0 / tempo  # seconds per beat

    # Calculate target measures from input duration
    total_beats = analysis.duration / spb
    target_measures = max(1, math.ceil(total_beats / beats_per_measure))
    score_duration = target_measures * beats_per_measure * spb

    pattern = _rock_pattern(target_measures, beats_per_measure)

    drums = Voice(
        notes=pattern,
        name="Drums",
        midi_program=0,
        clef="perc",
        midi_channel=10,
    )

    return Score(
        voices=[drums],
        tempo=tempo,
        key=analysis.key,
        time_sig=(4, 4),
        duration=score_duration,
    )
