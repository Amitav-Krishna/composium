"""
Test script for melody accuracy comparison.

Usage:
    cd voicebeat && python scripts/test_melody_accuracy.py

Compares generated melody output against target sample-song.m4a
"""

import asyncio
import sys
import json
from pathlib import Path
import numpy as np

# Allow imports from the voicebeat root
sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import soundfile as sf
from pydub import AudioSegment as PydubSegment

from app.services import pitch_analyzer, composium_bridge
from app.models.schemas import MelodyContour
from config.settings import settings


def load_audio(path: str, sr: int = 22050):
    """Load audio file to numpy array."""
    try:
        y, orig_sr = librosa.load(path, sr=sr, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None


def convert_to_wav_bytes(path: str) -> bytes:
    """Convert any audio file to WAV bytes using pydub."""
    import io
    audio = PydubSegment.from_file(path)
    audio = audio.set_frame_rate(22050).set_channels(1)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()


def analyze_pitch_contour(y, sr, fmin=130, fmax=800):
    """Extract pitch contour from audio."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr
    )
    # Filter out unvoiced frames
    voiced_indices = np.where(voiced_flag)[0]
    voiced_f0 = f0[voiced_indices]
    voiced_times = voiced_indices * (512 / sr)  # hop_length=512 default
    return voiced_times, voiced_f0, voiced_probs[voiced_indices] if voiced_probs is not None else None


def compare_pitch_contours(times1, f0_1, times2, f0_2):
    """Compare two pitch contours and return similarity metrics."""
    if len(f0_1) == 0 or len(f0_2) == 0:
        return {"error": "One or both contours are empty"}

    # Convert to MIDI for easier comparison
    midi1 = 69 + 12 * np.log2(f0_1 / 440.0)
    midi2 = 69 + 12 * np.log2(f0_2 / 440.0)

    # Basic stats
    stats = {
        "input_notes": len(midi1),
        "target_notes": len(midi2),
        "input_mean_midi": float(np.mean(midi1)),
        "target_mean_midi": float(np.mean(midi2)),
        "input_range": float(np.max(midi1) - np.min(midi1)),
        "target_range": float(np.max(midi2) - np.min(midi2)),
        "mean_diff": float(np.mean(midi1) - np.mean(midi2)),
    }

    # Pitch class distribution (ignoring octave)
    pc1 = np.round(midi1) % 12
    pc2 = np.round(midi2) % 12

    pc_hist1 = np.histogram(pc1, bins=12, range=(0, 12))[0]
    pc_hist2 = np.histogram(pc2, bins=12, range=(0, 12))[0]

    # Normalize
    pc_hist1 = pc_hist1 / np.sum(pc_hist1) if np.sum(pc_hist1) > 0 else pc_hist1
    pc_hist2 = pc_hist2 / np.sum(pc_hist2) if np.sum(pc_hist2) > 0 else pc_hist2

    # Cosine similarity of pitch class distribution
    dot = np.dot(pc_hist1, pc_hist2)
    norm1 = np.linalg.norm(pc_hist1)
    norm2 = np.linalg.norm(pc_hist2)
    stats["pitch_class_similarity"] = float(dot / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

    return stats


async def process_input_and_compare():
    """Main test function."""
    samples_dir = Path(__file__).parent.parent / "samples"
    input_file = samples_dir / "sample-input.m4a"
    target_file = samples_dir / "sample-song.m4a"

    print("=" * 70)
    print("MELODY ACCURACY TEST")
    print("=" * 70)

    # Check files exist
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    if not target_file.exists():
        print(f"ERROR: Target file not found: {target_file}")
        return

    print(f"\nInput file: {input_file}")
    print(f"Target file: {target_file}")

    # Load and analyze input
    print("\n--- Analyzing INPUT audio ---")
    y_input, sr = load_audio(str(input_file))
    if y_input is None:
        return

    print(f"Duration: {len(y_input)/sr:.2f}s")

    input_times, input_f0, input_conf = analyze_pitch_contour(
        y_input, sr, fmin=settings.pitch_fmin, fmax=settings.pitch_fmax
    )
    print(f"Detected {len(input_f0)} voiced frames")
    if len(input_f0) > 0:
        print(f"Pitch range: {librosa.hz_to_note(np.min(input_f0))} - {librosa.hz_to_note(np.max(input_f0))}")

    # Load and analyze target
    print("\n--- Analyzing TARGET audio ---")
    y_target, sr = load_audio(str(target_file))
    if y_target is None:
        return

    print(f"Duration: {len(y_target)/sr:.2f}s")

    target_times, target_f0, target_conf = analyze_pitch_contour(
        y_target, sr, fmin=80, fmax=2000  # Wider range for guitar
    )
    print(f"Detected {len(target_f0)} voiced frames")
    if len(target_f0) > 0:
        print(f"Pitch range: {librosa.hz_to_note(np.min(target_f0))} - {librosa.hz_to_note(np.max(target_f0))}")

    # Compare
    print("\n--- Comparison ---")
    comparison = compare_pitch_contours(input_times, input_f0, target_times, target_f0)
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Now run the pipeline on the input
    print("\n--- Running melody extraction pipeline ---")
    print(f"Current settings:")
    print(f"  pitch_fmin: {settings.pitch_fmin}")
    print(f"  pitch_fmax: {settings.pitch_fmax}")
    print(f"  min_note_duration: {settings.min_note_duration}")
    print(f"  onset_delta: {settings.onset_delta}")
    print(f"  pitch_shift: {settings.pitch_shift}")

    # Load input as WAV bytes for the pipeline (convert from m4a)
    input_bytes = convert_to_wav_bytes(str(input_file))

    # Run melody analysis
    melody = await pitch_analyzer.analyze_melody(input_bytes)

    print(f"\nExtracted melody:")
    print(f"  Notes detected: {len(melody.pitches)}")
    print(f"  Key signature: {melody.key_signature}")
    print(f"  Tempo: {melody.tempo_bpm} BPM")

    if melody.pitches:
        notes = [(p.note_name, p.duration_seconds) for p in melody.pitches[:10]]
        print(f"  First 10 notes: {notes}")

        midi_notes = [p.midi_note for p in melody.pitches]
        print(f"  MIDI range: {min(midi_notes)} - {max(midi_notes)}")

    # Render the melody
    print("\n--- Rendering melody with guitar ---")
    output_file = await composium_bridge.render_melody(
        melody,
        instrument="guitar",
        bpm=melody.tempo_bpm or 120,
        key=melody.key_signature,
    )
    print(f"Rendered to: {output_file}")

    # Analyze the output
    print("\n--- Analyzing OUTPUT audio ---")
    y_output, sr = load_audio(output_file)
    if y_output is None:
        print("Failed to load output")
        return

    print(f"Duration: {len(y_output)/sr:.2f}s")

    output_times, output_f0, output_conf = analyze_pitch_contour(
        y_output, sr, fmin=80, fmax=2000
    )
    print(f"Detected {len(output_f0)} voiced frames")
    if len(output_f0) > 0:
        print(f"Pitch range: {librosa.hz_to_note(np.min(output_f0))} - {librosa.hz_to_note(np.max(output_f0))}")

    # Compare output to target
    print("\n--- Output vs Target Comparison ---")
    output_comparison = compare_pitch_contours(output_times, output_f0, target_times, target_f0)
    for key, value in output_comparison.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    return melody, output_file, comparison, output_comparison


if __name__ == "__main__":
    asyncio.run(process_input_and_compare())
