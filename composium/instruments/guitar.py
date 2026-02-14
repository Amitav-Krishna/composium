"""Guitar-specific arrangement patterns."""

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


def _transpose_to_range(notes: list[Note], low: int = 40, high: int = 76) -> list[Note]:
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


def _fingerpick(chords: list, beats_per_measure: float = 4.0) -> list[Note]:
    """Generate fingerpicking arpeggios (root, third, fifth, octave) from chords."""
    notes: list[Note] = []
    grid = 0.5  # eighth notes

    for chord in chords:
        root = chord.root_midi
        # Clamp root into low guitar range (MIDI 40-64, E2-E4)
        while root < 40:
            root += 12
        while root > 64:
            root -= 12

        if chord.quality == "minor":
            third = root + 3
        elif chord.quality == "dim":
            third = root + 3
        else:
            third = root + 4
        fifth = root + 7
        octave = root + 12

        # Ascending arpeggio pattern: root, third, fifth, octave
        pattern = [root, third, fifth, octave]
        beat = chord.start_beat
        end_beat = chord.start_beat + chord.duration_beats

        i = 0
        while beat < end_beat - 0.01:
            midi = pattern[i % len(pattern)]
            notes.append(Note(midi, beat, grid))
            beat += grid
            i += 1

    return notes


def arrange_guitar(analysis: Analysis) -> Score:
    """Generate a 2-voice guitar Score from an Analysis."""
    tempo = analysis.tempo or 120
    beats_per_measure = 4.0  # 4/4 time
    spb = 60.0 / tempo  # seconds per beat

    # Calculate target measures from input duration
    total_beats = analysis.duration / spb
    target_measures = max(1, math.ceil(total_beats / beats_per_measure))
    score_duration = target_measures * beats_per_measure * spb

    # --- Voice 1: Lead melody ---
    melody = list(analysis.notes)
    melody = _transpose_to_range(melody, low=40, high=76)  # E2-E5
    melody = _quantize_to_grid(melody, grid=0.5)
    melody = fill_duration(melody, target_measures, beats_per_measure)

    # --- Chords from melody ---
    chords = generate_chords(melody, analysis.key, beats_per_measure)

    # --- Voice 2: Fingerpicking accompaniment ---
    fingerpicking = _fingerpick(chords, beats_per_measure)

    lead = Voice(notes=melody, name="Lead", midi_program=25, clef="treble")
    accomp = Voice(notes=fingerpicking, name="Fingerpicking", midi_program=25, clef="treble")

    return Score(
        voices=[lead, accomp],
        tempo=tempo,
        key=analysis.key,
        time_sig=(4, 4),
        duration=score_duration,
    )
