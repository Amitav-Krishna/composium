"""Piano-specific arrangement patterns."""

from __future__ import annotations

import math

from composium.notation import Analysis, Note, Score, Voice
from composium.arrangement import generate_chords, fill_duration


def _quantize_to_grid(notes: list[Note], grid: float = 0.5) -> list[Note]:
    """Snap note starts and durations to the nearest grid unit (default: 8th note)."""
    result: list[Note] = []
    for n in notes:
        start = round(n.start_beat / grid) * grid
        dur = max(grid, round(n.duration_beats / grid) * grid)
        result.append(Note(n.midi_pitch, start, dur))
    return result


def _transpose_to_range(notes: list[Note], low: int = 60, high: int = 84) -> list[Note]:
    """Transpose notes into [low, high] MIDI range, shifting by octaves."""
    if not notes:
        return []

    pitches = [n.midi_pitch for n in notes]
    median = sorted(pitches)[len(pitches) // 2]
    target_center = (low + high) // 2

    # Shift by whole octaves to center the melody
    shift = round((target_center - median) / 12) * 12

    result: list[Note] = []
    for n in notes:
        new_pitch = n.midi_pitch + shift
        # Clamp any outliers
        while new_pitch < low:
            new_pitch += 12
        while new_pitch > high:
            new_pitch -= 12
        result.append(Note(new_pitch, n.start_beat, n.duration_beats))
    return result


def _alberti_bass(chords: list, beats_per_measure: float = 4.0) -> list[Note]:
    """Generate Alberti bass arpeggios (root-fifth-third-fifth) from chords."""
    notes: list[Note] = []
    grid = 0.5  # eighth notes

    for chord in chords:
        root = chord.root_midi
        if chord.quality == "minor":
            third = root + 3
        elif chord.quality == "dim":
            third = root + 3
        else:
            third = root + 4
        fifth = root + 7

        # Alberti pattern: root, fifth, third, fifth (repeated)
        pattern = [root, fifth, third, fifth]
        beat = chord.start_beat
        end_beat = chord.start_beat + chord.duration_beats

        i = 0
        while beat < end_beat - 0.01:
            midi = pattern[i % len(pattern)]
            notes.append(Note(midi, beat, grid))
            beat += grid
            i += 1

    return notes


def arrange_piano(analysis: Analysis) -> Score:
    """Generate a 2-voice piano Score from an Analysis."""
    tempo = analysis.tempo or 120
    beats_per_measure = 4.0  # 4/4 time
    spb = 60.0 / tempo  # seconds per beat

    # Calculate target measures from input duration
    total_beats = analysis.duration / spb
    target_measures = max(1, math.ceil(total_beats / beats_per_measure))
    score_duration = target_measures * beats_per_measure * spb

    # --- Right hand: melody ---
    melody = list(analysis.notes)
    melody = _transpose_to_range(melody, low=60, high=84)  # C4-C6
    melody = _quantize_to_grid(melody, grid=0.5)
    melody = fill_duration(melody, target_measures, beats_per_measure)

    # --- Chords from melody ---
    chords = generate_chords(melody, analysis.key, beats_per_measure)

    # --- Left hand: Alberti bass ---
    bass = _alberti_bass(chords, beats_per_measure)

    rh = Voice(notes=melody, name="Right Hand", midi_program=0, clef="treble")
    lh = Voice(notes=bass, name="Left Hand", midi_program=0, clef="bass")

    return Score(
        voices=[rh, lh],
        tempo=tempo,
        key=analysis.key,
        time_sig=(4, 4),
        duration=score_duration,
    )
