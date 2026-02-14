"""EDM-specific arrangement with section-based structure.

Divides the track into three sections — Intro, Build, Drop — each with
distinct voices, velocities, and rhythmic patterns.
"""

from __future__ import annotations

import math

from composium.notation import Analysis, Note, Score, Voice
from composium.arrangement import generate_chords, fill_duration
from composium.instruments.drums import KICK, SNARE, HIHAT_CLOSED


# ---------------------------------------------------------------------------
# Section boundaries
# ---------------------------------------------------------------------------

def _section_boundaries(total_measures: int) -> tuple[int, int, int]:
    """Divide *total_measures* into intro/build/drop thirds.

    Returns ``(intro_end, build_end, drop_end)`` where each boundary is
    a measure index (0-based, exclusive).
    """
    third = max(1, round(total_measures / 3))
    intro_end = third
    build_end = min(intro_end + third, total_measures)
    drop_end = total_measures
    # Ensure each section has at least 1 measure
    if build_end <= intro_end:
        build_end = intro_end + 1
    if drop_end <= build_end:
        drop_end = build_end + 1
    return intro_end, build_end, drop_end


# ---------------------------------------------------------------------------
# Chord helpers
# ---------------------------------------------------------------------------

def _voice_chord(root_midi: int, quality: str) -> list[int]:
    """Return root + third + fifth MIDI pitches for a triad, voiced in C4-G5."""
    # Transpose root into C4-G5 range (MIDI 60-79)
    root = root_midi
    while root < 60:
        root += 12
    while root > 72:
        root -= 12
    third_interval = 3 if quality in ("minor", "dim") else 4
    fifth_interval = 6 if quality == "dim" else 7
    return [root, root + third_interval, root + fifth_interval]


def _pad_chords(
    chords: list,
    start_m: int,
    end_m: int,
    pattern: str,
    beats_per_measure: float,
) -> list[Note]:
    """Generate pad chord notes for measures [start_m, end_m).

    *pattern*: ``"whole"`` (one chord per measure), ``"half"`` (two per
    measure), or ``"stab"`` (quarter notes on beats 1 & 3).
    """
    notes: list[Note] = []
    for m_idx in range(start_m, end_m):
        if m_idx >= len(chords):
            break
        chord = chords[m_idx]
        pitches = _voice_chord(chord.root_midi, chord.quality)
        m_start = m_idx * beats_per_measure

        if pattern == "whole":
            for p in pitches:
                notes.append(Note(p, m_start, beats_per_measure))
        elif pattern == "half":
            half = beats_per_measure / 2
            for p in pitches:
                notes.append(Note(p, m_start, half))
                notes.append(Note(p, m_start + half, half))
        elif pattern == "stab":
            for beat_offset in (0.0, 2.0):
                for p in pitches:
                    notes.append(Note(p, m_start + beat_offset, 1.0))
    return notes


# ---------------------------------------------------------------------------
# Bass helpers
# ---------------------------------------------------------------------------

def _synth_bass(
    chords: list,
    start_m: int,
    end_m: int,
    pattern: str,
    beats_per_measure: float,
) -> list[Note]:
    """Generate synth bass notes for measures [start_m, end_m).

    *pattern*: ``"half"`` (half-note root) or ``"eighth"`` (eighth-note pulse).
    """
    notes: list[Note] = []
    for m_idx in range(start_m, end_m):
        if m_idx >= len(chords):
            break
        chord = chords[m_idx]
        # Bass in octave 2-3 (MIDI 36-48)
        root = chord.root_midi
        while root < 36:
            root += 12
        while root > 48:
            root -= 12
        m_start = m_idx * beats_per_measure

        if pattern == "half":
            half = beats_per_measure / 2
            notes.append(Note(root, m_start, half))
            notes.append(Note(root, m_start + half, half))
        elif pattern == "eighth":
            grid = 0.5
            beat = m_start
            end_beat = m_start + beats_per_measure
            while beat < end_beat - 0.01:
                notes.append(Note(root, beat, grid))
                beat += grid
    return notes


# ---------------------------------------------------------------------------
# Drum helpers
# ---------------------------------------------------------------------------

def _edm_drums_build(
    start_m: int,
    end_m: int,
    beats_per_measure: float,
) -> list[Note]:
    """Hi-hats only, plus snare roll in the last 2 measures of the build."""
    notes: list[Note] = []
    grid = 0.5
    snare_roll_start = max(start_m, end_m - 2)

    for m_idx in range(start_m, end_m):
        m_start = m_idx * beats_per_measure
        in_snare_roll = m_idx >= snare_roll_start

        for slot in range(int(beats_per_measure / grid)):
            beat = m_start + slot * grid
            notes.append(Note(HIHAT_CLOSED, beat, grid))
            if in_snare_roll:
                notes.append(Note(SNARE, beat, grid))

    return notes


def _edm_drums_drop(
    start_m: int,
    end_m: int,
    beats_per_measure: float,
) -> list[Note]:
    """Four-on-the-floor pattern.

    Beat:  1   1+  2   2+  3   3+  4   4+
    Kick:  X       X       X       X
    Snare:         X               X
    HH:    X   X   X   X   X   X   X   X
    """
    notes: list[Note] = []
    grid = 0.5

    for m_idx in range(start_m, end_m):
        m_start = m_idx * beats_per_measure
        slots = int(beats_per_measure / grid)

        for slot in range(slots):
            beat = m_start + slot * grid
            # Hi-hat on every eighth note
            notes.append(Note(HIHAT_CLOSED, beat, grid))
            # Kick on every beat (slots 0, 2, 4, 6)
            if slot % 2 == 0:
                notes.append(Note(KICK, beat, grid))
            # Snare on beats 2 and 4 (slots 2, 6)
            if slot == 2 or slot == 6:
                notes.append(Note(SNARE, beat, grid))

    return notes


# ---------------------------------------------------------------------------
# Main arrangement
# ---------------------------------------------------------------------------

def arrange_edm(analysis: Analysis) -> Score:
    """Generate a section-aware EDM Score from an Analysis."""
    tempo = analysis.tempo or 120
    beats_per_measure = 4.0
    spb = 60.0 / tempo

    total_beats = analysis.duration / spb
    target_measures = max(3, math.ceil(total_beats / beats_per_measure))
    score_duration = target_measures * beats_per_measure * spb

    intro_end, build_end, drop_end = _section_boundaries(target_measures)

    # Generate chords from melody
    melody = fill_duration(list(analysis.notes), target_measures, beats_per_measure)
    chords = generate_chords(melody, analysis.key, beats_per_measure)

    # --- Pad voices (one per section) ---
    pad_intro_notes = _pad_chords(chords, 0, intro_end, "whole", beats_per_measure)
    pad_build_notes = _pad_chords(chords, intro_end, build_end, "half", beats_per_measure)
    pad_drop_notes = _pad_chords(chords, build_end, drop_end, "stab", beats_per_measure)

    pad_intro = Voice(
        notes=pad_intro_notes, name="Pad Intro",
        midi_program=4, velocity=50,
    )
    pad_build = Voice(
        notes=pad_build_notes, name="Pad Build",
        midi_program=4, velocity=75,
    )
    pad_drop = Voice(
        notes=pad_drop_notes, name="Pad Drop",
        midi_program=4, velocity=110,
    )

    # --- Bass voices ---
    bass_build_notes = _synth_bass(chords, intro_end, build_end, "half", beats_per_measure)
    bass_drop_notes = _synth_bass(chords, build_end, drop_end, "eighth", beats_per_measure)

    bass_build = Voice(
        notes=bass_build_notes, name="Bass Build",
        midi_program=38, velocity=70,
    )
    bass_drop = Voice(
        notes=bass_drop_notes, name="Bass Drop",
        midi_program=38, velocity=115,
    )

    # --- Drum voices ---
    drums_build_notes = _edm_drums_build(intro_end, build_end, beats_per_measure)
    drums_drop_notes = _edm_drums_drop(build_end, drop_end, beats_per_measure)

    drums_build = Voice(
        notes=drums_build_notes, name="Drums Build",
        midi_program=0, clef="perc", midi_channel=10, velocity=60,
    )
    drums_drop = Voice(
        notes=drums_drop_notes, name="Drums Drop",
        midi_program=0, clef="perc", midi_channel=10, velocity=120,
    )

    return Score(
        voices=[
            pad_intro, pad_build, pad_drop,
            bass_build, bass_drop,
            drums_build, drums_drop,
        ],
        tempo=tempo,
        key=analysis.key,
        time_sig=(4, 4),
        duration=score_duration,
    )


# ---------------------------------------------------------------------------
# Humming envelope for vocal layering
# ---------------------------------------------------------------------------

def edm_humming_envelope(analysis: Analysis) -> list[tuple[float, float]]:
    """Compute a volume envelope for the raw humming audio.

    Returns a list of ``(time_sec, volume)`` breakpoints:
    - Intro: 0.3 (soft)
    - Build: ramp 0.3 → 0.8
    - Drop: 1.0 (full)
    """
    tempo = analysis.tempo or 120
    beats_per_measure = 4.0
    spb = 60.0 / tempo

    total_beats = analysis.duration / spb
    target_measures = max(3, math.ceil(total_beats / beats_per_measure))

    intro_end, build_end, drop_end = _section_boundaries(target_measures)

    intro_end_sec = intro_end * beats_per_measure * spb
    build_end_sec = build_end * beats_per_measure * spb
    drop_end_sec = drop_end * beats_per_measure * spb

    return [
        (0.0, 0.3),                # start of intro
        (intro_end_sec, 0.3),      # end of intro / start of build
        (build_end_sec, 0.8),      # end of build (ramped up)
        (build_end_sec + 0.01, 1.0),  # start of drop
        (drop_end_sec, 1.0),       # end
    ]
