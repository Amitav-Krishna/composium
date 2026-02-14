"""
Music Theory Utilities

Key detection, scale quantization, duration normalization, and other music theory helpers.
"""

import numpy as np
from typing import Optional

# Note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale intervals (semitones from root)
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}

# Standard note durations in beats
STANDARD_DURATIONS = [
    0.125,  # 32nd note
    0.25,   # 16th note
    0.5,    # 8th note
    0.75,   # dotted 8th
    1.0,    # quarter note
    1.5,    # dotted quarter
    2.0,    # half note
    3.0,    # dotted half
    4.0,    # whole note
]


def get_scale_notes(root: str, scale_type: str = "major") -> list[int]:
    """
    Get MIDI note numbers for a scale starting at the given root.

    Args:
        root: Root note (e.g., "C", "F#", "Bb")
        scale_type: Type of scale (major, minor, etc.)

    Returns:
        List of MIDI note numbers for one octave of the scale (starting at octave 4)
    """
    root_midi = note_name_to_midi(root + "4")
    intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])

    return [root_midi + interval for interval in intervals]


def note_name_to_midi(note: str) -> int:
    """
    Convert note name to MIDI number.

    Args:
        note: Note name like "C4", "F#5", "Bb3"

    Returns:
        MIDI note number
    """
    # Parse note name
    note = note.strip()

    # Handle flats by converting to sharps
    note = note.replace("Db", "C#").replace("Eb", "D#").replace("Fb", "E")
    note = note.replace("Gb", "F#").replace("Ab", "G#").replace("Bb", "A#")

    # Extract note and octave
    if len(note) >= 2 and note[1] == "#":
        note_name = note[:2]
        octave_str = note[2:]
    else:
        note_name = note[0]
        octave_str = note[1:]

    try:
        octave = int(octave_str)
    except ValueError:
        octave = 4  # Default octave

    # Find note index
    note_index = NOTE_NAMES.index(note_name) if note_name in NOTE_NAMES else 0

    return (octave + 1) * 12 + note_index


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name."""
    octave = (midi // 12) - 1
    note_index = midi % 12
    return f"{NOTE_NAMES[note_index]}{octave}"


def quantize_to_scale(
    midi_notes: list[int],
    root: str,
    scale_type: str = "major",
) -> list[int]:
    """
    Snap MIDI notes to the nearest note in the given scale.

    Args:
        midi_notes: List of MIDI note numbers
        root: Root note of the scale
        scale_type: Type of scale

    Returns:
        List of quantized MIDI notes
    """
    root_midi = note_name_to_midi(root + "0") % 12
    intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])

    # Build set of valid pitch classes
    valid_pcs = set((root_midi + interval) % 12 for interval in intervals)

    quantized = []
    for midi in midi_notes:
        pc = midi % 12
        octave = midi // 12

        if pc in valid_pcs:
            quantized.append(midi)
        else:
            # Find nearest valid pitch class
            best_distance = 12
            best_pc = pc
            for vpc in valid_pcs:
                dist = min(abs(vpc - pc), 12 - abs(vpc - pc))
                if dist < best_distance:
                    best_distance = dist
                    best_pc = vpc

            # Choose direction that stays closest
            if (pc - best_pc) % 12 <= 6:
                new_midi = octave * 12 + best_pc
            else:
                new_midi = octave * 12 + best_pc

            quantized.append(new_midi)

    return quantized


def quantize_duration(
    duration_seconds: float,
    bpm: int,
    grid_subdivision: int = 16,
) -> float:
    """
    Quantize a duration to the nearest grid position.

    Args:
        duration_seconds: Duration in seconds
        bpm: Tempo in BPM
        grid_subdivision: Grid subdivision (16 = 16th notes, 8 = 8th notes)

    Returns:
        Quantized duration in beats
    """
    beats_per_second = bpm / 60.0
    duration_beats = duration_seconds * beats_per_second

    # Quantize to grid
    grid_size = 4.0 / grid_subdivision  # e.g., 0.25 beats for 16th notes
    quantized_beats = round(duration_beats / grid_size) * grid_size

    # Minimum of one grid unit
    return max(grid_size, quantized_beats)


def quantize_to_standard_duration(duration_beats: float) -> float:
    """
    Snap a duration to the nearest standard musical duration.

    Args:
        duration_beats: Duration in beats

    Returns:
        Nearest standard duration
    """
    best_duration = STANDARD_DURATIONS[0]
    best_distance = abs(duration_beats - best_duration)

    for std_dur in STANDARD_DURATIONS[1:]:
        distance = abs(duration_beats - std_dur)
        if distance < best_distance:
            best_distance = distance
            best_duration = std_dur

    return best_duration


def detect_key_from_pitch_classes(
    pitch_classes: list[int],
    weights: Optional[list[float]] = None,
) -> tuple[str, str]:
    """
    Detect the most likely key from a list of pitch classes.

    Args:
        pitch_classes: List of pitch classes (0-11)
        weights: Optional weights for each pitch class

    Returns:
        Tuple of (root note name, scale type)
    """
    if not pitch_classes:
        return ("C", "major")

    # Build weighted histogram
    histogram = np.zeros(12)
    for i, pc in enumerate(pitch_classes):
        weight = weights[i] if weights else 1.0
        histogram[pc % 12] += weight

    # Normalize
    total = np.sum(histogram)
    if total > 0:
        histogram /= total

    # Templates for major and minor
    major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                               5.19, 2.39, 3.66, 2.29, 2.88])  # Krumhansl-Kessler
    minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                               4.75, 3.98, 2.69, 3.34, 3.17])

    # Normalize templates
    major_template /= np.sum(major_template)
    minor_template /= np.sum(minor_template)

    best_root = 0
    best_mode = "major"
    best_score = -1

    for root in range(12):
        # Rotate histogram to align with template
        rotated = np.roll(histogram, -root)

        major_score = np.corrcoef(rotated, major_template)[0, 1]
        minor_score = np.corrcoef(rotated, minor_template)[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_root = root
            best_mode = "major"

        if minor_score > best_score:
            best_score = minor_score
            best_root = root
            best_mode = "minor"

    return (NOTE_NAMES[best_root], best_mode)


def get_chord_for_scale_degree(
    degree: int,
    root: str,
    scale_type: str = "major",
) -> list[int]:
    """
    Get the chord (triad) for a scale degree.

    Args:
        degree: Scale degree (1-7)
        root: Root note of the key
        scale_type: Type of scale

    Returns:
        List of MIDI note numbers forming the chord (starting at octave 4)
    """
    scale_notes = get_scale_notes(root, scale_type)

    # Get the root of the chord (0-indexed degree)
    chord_root_index = (degree - 1) % len(scale_notes)
    chord_third_index = (chord_root_index + 2) % len(scale_notes)
    chord_fifth_index = (chord_root_index + 4) % len(scale_notes)

    # Handle octave wrapping
    chord_root = scale_notes[chord_root_index]
    chord_third = scale_notes[chord_third_index]
    chord_fifth = scale_notes[chord_fifth_index]

    # Ensure ascending order
    while chord_third < chord_root:
        chord_third += 12
    while chord_fifth < chord_third:
        chord_fifth += 12

    return [chord_root, chord_third, chord_fifth]


def seconds_to_beats(seconds: float, bpm: int) -> float:
    """Convert seconds to beats."""
    return seconds * (bpm / 60.0)


def beats_to_seconds(beats: float, bpm: int) -> float:
    """Convert beats to seconds."""
    return beats * (60.0 / bpm)


def get_bar_and_position(
    time_beats: float,
    beats_per_bar: int = 4,
    subdivisions: int = 16,
) -> tuple[int, int]:
    """
    Convert a time in beats to bar number and position within bar.

    Args:
        time_beats: Time in beats
        beats_per_bar: Beats per bar (4 for 4/4 time)
        subdivisions: Subdivisions per bar (16 for 16th notes)

    Returns:
        Tuple of (bar number, position within bar)
    """
    subdivisions_per_beat = subdivisions / beats_per_bar
    total_subdivisions = int(time_beats * subdivisions_per_beat)

    bar = total_subdivisions // subdivisions
    position = total_subdivisions % subdivisions

    return (bar, position)
