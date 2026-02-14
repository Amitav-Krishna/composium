"""Audio analysis — tempo, pitch, key, and beat detection."""

from __future__ import annotations

import numpy as np
import librosa

from composium.notation import Analysis, Note


# ---------------------------------------------------------------------------
# Key detection templates (Krumhansl–Kessler profiles, simplified)
# ---------------------------------------------------------------------------

_MAJOR_TEMPLATE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
_MINOR_TEMPLATE = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)

_NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Prefer the conventional name for each key
_MAJOR_KEY_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MINOR_KEY_NAMES = ["Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"]


def _detect_key(pitch_classes: np.ndarray) -> str:
    """Detect key from an array of pitch classes (0-11) via template correlation."""
    histogram = np.zeros(12)
    for pc in pitch_classes:
        histogram[int(pc) % 12] += 1

    if histogram.sum() == 0:
        return "C"

    histogram = histogram / histogram.sum()

    best_score = -999.0
    best_key = "C"

    for root in range(12):
        rotated = np.roll(histogram, -root)
        major_score = float(np.corrcoef(rotated, _MAJOR_TEMPLATE)[0, 1])
        minor_score = float(np.corrcoef(rotated, _MINOR_TEMPLATE)[0, 1])

        if major_score > best_score:
            best_score = major_score
            best_key = _MAJOR_KEY_NAMES[root]
        if minor_score > best_score:
            best_score = minor_score
            best_key = _MINOR_KEY_NAMES[root]

    return best_key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(audio_path: str) -> Analysis:
    """Analyze an audio file and return tempo, key, notes, beats, and duration."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Tempo and beats
    tempo_val, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # librosa may return an ndarray for tempo
    tempo = int(round(float(np.asarray(tempo_val).flat[0])))
    beat_times = list(librosa.frames_to_time(beat_frames, sr=sr).astype(float))

    # Detect onsets (note attacks) to split notes even when pitch stays the same
    # This helps capture "da da da" patterns where pitch is constant
    # Use lower delta threshold for more sensitive peak detection
    hop_length = 512
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True, delta=0.05
    )
    onset_frame_set = set(onset_frames.tolist())

    # Pitch tracking via pyin (good for monophonic humming/singing)
    # Use fmin=130Hz (C3) to capture vocal range while filtering bass rumble
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=130, fmax=800, sr=sr, frame_length=2048, hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Convert to MIDI, quantize, segment into Note events
    notes: list[Note] = []
    pitch_classes: list[int] = []

    # Seconds per beat
    spb = 60.0 / tempo if tempo > 0 else 0.5

    prev_midi: int | None = None
    note_start_time: float = 0.0

    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        t = float(times[i])
        if voiced and not np.isnan(freq) and freq > 0:
            midi_float = float(librosa.hz_to_midi(freq))
            midi_int = int(round(midi_float))
            pitch_classes.append(midi_int % 12)

            # Check if this frame is an onset (new note attack)
            is_onset = i in onset_frame_set and prev_midi is not None

            if midi_int != prev_midi or is_onset:
                # Close previous note (pitch changed OR onset detected)
                if prev_midi is not None:
                    dur_sec = t - note_start_time
                    dur_beats = dur_sec / spb
                    start_beat = note_start_time / spb
                    if dur_beats > 0.02:
                        notes.append(Note(prev_midi, start_beat, dur_beats))
                prev_midi = midi_int
                note_start_time = t
        else:
            # Silence — close any open note
            if prev_midi is not None:
                dur_sec = t - note_start_time
                dur_beats = dur_sec / spb
                start_beat = note_start_time / spb
                if dur_beats > 0.02:
                    notes.append(Note(prev_midi, start_beat, dur_beats))
                prev_midi = None

    # Close final note
    if prev_midi is not None:
        dur_sec = duration - note_start_time
        dur_beats = dur_sec / spb
        start_beat = note_start_time / spb
        if dur_beats > 0.02:
            notes.append(Note(prev_midi, start_beat, dur_beats))

    key = _detect_key(np.array(pitch_classes)) if pitch_classes else "C"

    return Analysis(
        tempo=tempo,
        key=key,
        duration=duration,
        notes=notes,
        beats=beat_times,
    )
