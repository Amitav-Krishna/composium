"""
Pitch Analyzer Service

Extracts melody/pitch from hummed or sung audio segments using librosa.pyin.
Converts to note events and ABC notation.
"""

import io
import numpy as np
import librosa
import soundfile as sf
from typing import Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schemas import MelodyContour, PitchEvent


# Note names for MIDI conversion
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ABC notation note mapping (for octave 4)
ABC_NOTES = {
    "C": "C", "C#": "^C", "D": "D", "D#": "^D", "E": "E", "F": "F",
    "F#": "^F", "G": "G", "G#": "^G", "A": "A", "A#": "^A", "B": "B"
}


async def analyze_melody(
    audio_bytes: bytes,
    sr: int = 22050,
    fmin: float = 80,
    fmax: float = 800,
    hop_length: int = 512,
    min_note_duration: float = 0.1,
) -> MelodyContour:
    """
    Analyze a melody segment and extract pitch events.

    Args:
        audio_bytes: Raw audio bytes of the melody segment
        sr: Sample rate to use
        fmin: Minimum frequency to detect
        fmax: Maximum frequency to detect
        hop_length: Hop length for analysis
        min_note_duration: Minimum duration (seconds) for a note event

    Returns:
        MelodyContour with pitch events, key signature, and ABC notation
    """
    # Load audio
    y = _load_audio(audio_bytes, sr)

    # Extract pitch using pyin (better for voice than basic pitch detection)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )

    # Convert to pitch events
    pitches = _extract_pitch_events(f0, voiced_flag, voiced_probs, sr, hop_length, min_note_duration)

    # Detect key signature
    key_sig = _detect_key(pitches) if pitches else None

    # Estimate tempo from the melody
    tempo = _estimate_tempo_from_pitches(pitches) if pitches else None

    # Generate ABC notation
    abc = _generate_abc_notation(pitches, key_sig, tempo) if pitches else None

    return MelodyContour(
        pitches=pitches,
        key_signature=key_sig,
        abc_notation=abc,
        tempo_bpm=tempo,
    )


def _load_audio(audio_bytes: bytes, target_sr: int = 22050) -> np.ndarray:
    """Load audio from bytes."""
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y.astype(np.float32)
    except Exception:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        return y


def _extract_pitch_events(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
    sr: int,
    hop_length: int,
    min_duration: float,
) -> list[PitchEvent]:
    """Convert frame-wise pitch data to note events."""
    if f0 is None or len(f0) == 0:
        return []

    events = []
    frame_duration = hop_length / sr

    # Group consecutive frames with the same pitch into notes
    current_note = None
    current_start = None
    current_frames = 0
    current_confidence = []

    for i, (freq, voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_probs)):
        time = i * frame_duration

        if voiced and freq > 0:
            midi = _freq_to_midi(freq)
            rounded_midi = round(midi)

            if current_note is None:
                # Start new note
                current_note = rounded_midi
                current_start = time
                current_frames = 1
                current_confidence = [prob if prob else 1.0]
            elif rounded_midi == current_note:
                # Continue current note
                current_frames += 1
                current_confidence.append(prob if prob else 1.0)
            else:
                # Different note - save current and start new
                duration = current_frames * frame_duration
                if duration >= min_duration:
                    events.append(_create_pitch_event(
                        current_start, current_note, duration, current_confidence
                    ))

                current_note = rounded_midi
                current_start = time
                current_frames = 1
                current_confidence = [prob if prob else 1.0]
        else:
            # Unvoiced - save current note if any
            if current_note is not None:
                duration = current_frames * frame_duration
                if duration >= min_duration:
                    events.append(_create_pitch_event(
                        current_start, current_note, duration, current_confidence
                    ))
                current_note = None
                current_start = None
                current_frames = 0
                current_confidence = []

    # Don't forget last note
    if current_note is not None:
        duration = current_frames * frame_duration
        if duration >= min_duration:
            events.append(_create_pitch_event(
                current_start, current_note, duration, current_confidence
            ))

    return events


def _create_pitch_event(
    start_time: float,
    midi_note: int,
    duration: float,
    confidences: list[float],
) -> PitchEvent:
    """Create a PitchEvent from note data."""
    freq = _midi_to_freq(midi_note)
    note_name = _midi_to_note_name(midi_note)
    avg_confidence = np.mean(confidences) if confidences else 1.0

    return PitchEvent(
        time_seconds=start_time,
        frequency_hz=freq,
        midi_note=midi_note,
        note_name=note_name,
        duration_seconds=duration,
        confidence=float(avg_confidence),
    )


def _freq_to_midi(freq: float) -> float:
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)


def _midi_to_freq(midi: int) -> float:
    """Convert MIDI note to frequency."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _midi_to_note_name(midi: int) -> str:
    """Convert MIDI note to note name (e.g., 'C4', 'A#3')."""
    octave = (midi // 12) - 1
    note_index = midi % 12
    return f"{NOTE_NAMES[note_index]}{octave}"


def _detect_key(pitches: list[PitchEvent]) -> Optional[str]:
    """
    Detect the most likely key signature from pitch events.

    Uses pitch class histogram correlation with major and minor scale templates.
    """
    if not pitches:
        return None

    # Count pitch classes (weighted by duration)
    pitch_class_counts = np.zeros(12)
    for p in pitches:
        pc = p.midi_note % 12
        pitch_class_counts[pc] += p.duration_seconds

    # Normalize
    total = np.sum(pitch_class_counts)
    if total == 0:
        return None
    pitch_class_counts /= total

    # Major and minor scale templates
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # Ionian
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # Aeolian

    # Normalize templates
    major_template = major_template / np.sum(major_template)
    minor_template = minor_template / np.sum(minor_template)

    best_key = None
    best_score = -1

    for root in range(12):
        # Rotate templates to each possible root
        major_rotated = np.roll(major_template, root)
        minor_rotated = np.roll(minor_template, root)

        # Correlation (dot product)
        major_score = np.dot(pitch_class_counts, major_rotated)
        minor_score = np.dot(pitch_class_counts, minor_rotated)

        if major_score > best_score:
            best_score = major_score
            best_key = f"{NOTE_NAMES[root]} major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{NOTE_NAMES[root]} minor"

    return best_key


def _estimate_tempo_from_pitches(pitches: list[PitchEvent]) -> Optional[int]:
    """Estimate tempo from note onsets."""
    if len(pitches) < 2:
        return None

    # Get inter-onset intervals
    onsets = [p.time_seconds for p in pitches]
    intervals = np.diff(onsets)

    if len(intervals) == 0:
        return None

    # Median interval as beat duration
    median_interval = np.median(intervals)

    if median_interval > 0:
        # Assume quarter notes
        bpm = 60.0 / median_interval
        # Clamp to reasonable range
        bpm = max(60, min(200, bpm))
        return int(round(bpm))

    return 120  # Default


def _generate_abc_notation(
    pitches: list[PitchEvent],
    key_sig: Optional[str],
    tempo: Optional[int],
) -> str:
    """Generate ABC notation string from pitch events."""
    if not pitches:
        return ""

    lines = [
        "X:1",
        "T:User Melody",
        f"M:4/4",
        f"L:1/8",
        f"Q:1/4={tempo or 120}",
        f"K:{_key_to_abc(key_sig)}",
    ]

    # Convert pitches to ABC notes
    abc_notes = []
    for p in pitches:
        abc_note = _pitch_to_abc(p)
        abc_notes.append(abc_note)

    # Join notes (simple approach - no barlines for now)
    melody_line = " ".join(abc_notes)
    lines.append(melody_line + " |]")

    return "\n".join(lines)


def _key_to_abc(key_sig: Optional[str]) -> str:
    """Convert key signature to ABC format."""
    if not key_sig:
        return "C"

    parts = key_sig.split()
    if len(parts) >= 2:
        note = parts[0].replace("#", "^").replace("b", "_")
        mode = parts[1].lower()
        if mode == "major":
            return note
        elif mode == "minor":
            return note + "m"
    return "C"


def _pitch_to_abc(pitch: PitchEvent) -> str:
    """Convert a pitch event to ABC notation."""
    midi = pitch.midi_note
    octave = (midi // 12) - 1
    note_index = midi % 12
    note_name = NOTE_NAMES[note_index]

    # ABC uses different notation for octaves
    # Octave 4 = normal (C D E F G A B)
    # Octave 5 = lowercase (c d e f g a b)
    # Octave 3 = C, D, etc. with comma
    # Octave 6 = c' d' etc.

    if octave <= 3:
        abc = ABC_NOTES.get(note_name, note_name)
        abc = abc + "," * (4 - octave)
    elif octave == 4:
        abc = ABC_NOTES.get(note_name, note_name)
    elif octave == 5:
        abc = ABC_NOTES.get(note_name, note_name).lower()
    else:
        abc = ABC_NOTES.get(note_name, note_name).lower()
        abc = abc + "'" * (octave - 5)

    # Add duration (simplified - using eighth notes as base)
    # Duration > 0.25s = quarter note (2), > 0.5s = half note (4)
    duration = pitch.duration_seconds
    if duration >= 0.5:
        abc += "4"
    elif duration >= 0.25:
        abc += "2"
    # else default eighth note (no modifier)

    return abc
