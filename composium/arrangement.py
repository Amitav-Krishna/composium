"""Arrangement engine — chord generation and duration filling."""

from __future__ import annotations

import math

from composium.notation import Note, Chord


# ---------------------------------------------------------------------------
# Scale / chord data
# ---------------------------------------------------------------------------

# Scale intervals (semitones from root) for major and natural minor
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# Diatonic triads built on each scale degree: (interval from degree root, quality)
_MAJOR_TRIADS = [
    (0, 4, 7, "major"),   # I
    (0, 3, 7, "minor"),   # ii
    (0, 3, 7, "minor"),   # iii
    (0, 4, 7, "major"),   # IV
    (0, 4, 7, "major"),   # V
    (0, 3, 7, "minor"),   # vi
    (0, 3, 6, "dim"),     # vii°
]
_MINOR_TRIADS = [
    (0, 3, 7, "minor"),   # i
    (0, 3, 6, "dim"),     # ii°
    (0, 4, 7, "major"),   # III
    (0, 3, 7, "minor"),   # iv
    (0, 3, 7, "minor"),   # v (natural minor)
    (0, 4, 7, "major"),   # VI
    (0, 4, 7, "major"),   # VII
]

# Root pitch class for key names
_KEY_ROOT: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    # Minor keys
    "Cm": 0, "C#m": 1, "Dm": 2, "Ebm": 3, "Em": 4, "Fm": 5,
    "F#m": 6, "Gm": 7, "G#m": 8, "Am": 9, "Bbm": 10, "Bm": 11,
}


def _is_minor(key: str) -> bool:
    return key.endswith("m")


# ---------------------------------------------------------------------------
# Chord generation
# ---------------------------------------------------------------------------

def generate_chords(
    notes: list[Note],
    key: str,
    beats_per_measure: float = 4.0,
) -> list[Chord]:
    """Infer a chord progression from melody notes using scale-degree heuristics."""
    if not notes:
        return []

    root_pc = _KEY_ROOT.get(key, 0)
    is_minor = _is_minor(key)
    scale = _MINOR_SCALE if is_minor else _MAJOR_SCALE
    triads = _MINOR_TRIADS if is_minor else _MAJOR_TRIADS

    # Figure out total measures
    max_beat = max(n.start_beat + n.duration_beats for n in notes)
    total_measures = max(1, math.ceil(max_beat / beats_per_measure))

    chords: list[Chord] = []

    for m_idx in range(total_measures):
        m_start = m_idx * beats_per_measure
        m_end = m_start + beats_per_measure

        # Pitch classes present in this measure (weighted by duration)
        pc_weight: dict[int, float] = {}
        for n in notes:
            if n.start_beat + n.duration_beats > m_start and n.start_beat < m_end:
                overlap = min(n.start_beat + n.duration_beats, m_end) - max(n.start_beat, m_start)
                pc = n.midi_pitch % 12
                pc_weight[pc] = pc_weight.get(pc, 0) + overlap

        if not pc_weight:
            # No melody in this measure — repeat last chord or use tonic
            if chords:
                prev = chords[-1]
                chords.append(Chord(prev.root_midi, prev.quality, m_start, beats_per_measure))
            else:
                chords.append(Chord(root_pc + 48, "minor" if is_minor else "major", m_start, beats_per_measure))
            continue

        # Score each diatonic triad
        best_score = -1.0
        best_degree = 0

        for deg_idx, (i0, i1, i2, quality) in enumerate(triads):
            chord_pcs = {
                (root_pc + scale[deg_idx] + i0) % 12,
                (root_pc + scale[deg_idx] + i1) % 12,
                (root_pc + scale[deg_idx] + i2) % 12,
            }
            score = sum(pc_weight.get(pc, 0) for pc in chord_pcs)
            if score > best_score:
                best_score = score
                best_degree = deg_idx

        deg_root_pc = (root_pc + scale[best_degree]) % 12
        _, _, _, quality = triads[best_degree]
        # Place chord root in bass octave (MIDI ~36-48)
        chord_root_midi = 36 + deg_root_pc
        if chord_root_midi < 36:
            chord_root_midi += 12

        chords.append(Chord(chord_root_midi, quality, m_start, beats_per_measure))

    return chords


# ---------------------------------------------------------------------------
# Duration filling
# ---------------------------------------------------------------------------

def fill_duration(
    melody_notes: list[Note],
    target_measures: int,
    beats_per_measure: float = 4.0,
) -> list[Note]:
    """Pad or repeat the melody to fill *target_measures* measures."""
    if not melody_notes:
        return []

    max_beat = max(n.start_beat + n.duration_beats for n in melody_notes)
    melody_length = max_beat  # in beats

    if melody_length <= 0:
        return list(melody_notes)

    target_beats = target_measures * beats_per_measure
    result: list[Note] = []
    offset = 0.0

    while offset < target_beats:
        for n in melody_notes:
            new_start = n.start_beat + offset
            if new_start >= target_beats:
                break
            new_dur = min(n.duration_beats, target_beats - new_start)
            if new_dur > 0:
                result.append(Note(n.midi_pitch, new_start, new_dur))
        offset += melody_length

    return result
