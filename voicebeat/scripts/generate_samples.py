#!/usr/bin/env python3
"""Generate placeholder audio samples for testing."""

import numpy as np
from pathlib import Path
import wave
import struct

# Sample configuration
SAMPLE_RATE = 44100
GENRES = ["hip-hop", "pop", "rock", "lo-fi", "electronic", "jazz", "rnb"]
INSTRUMENTS = {
    "kick": {"freq": 60, "duration": 0.2, "decay": 0.15},
    "snare": {"freq": 200, "duration": 0.15, "noise": True},
    "hi-hat": {"freq": 8000, "duration": 0.05, "noise": True},
    "piano": {"freq": 440, "duration": 0.5, "decay": 0.4},
    "bass": {"freq": 80, "duration": 0.3, "decay": 0.25},
    "guitar": {"freq": 330, "duration": 0.4, "decay": 0.35},
    "synth": {"freq": 523, "duration": 0.3, "decay": 0.2},
    "clap": {"freq": 1500, "duration": 0.1, "noise": True},
    "cymbal": {"freq": 6000, "duration": 0.3, "noise": True},
    "strings": {"freq": 392, "duration": 0.5, "decay": 0.45},
}


def generate_sine_wave(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * freq * t)


def generate_noise(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate white noise."""
    return np.random.uniform(-1, 1, int(sample_rate * duration))


def apply_decay(signal: np.ndarray, decay_time: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply exponential decay envelope."""
    t = np.linspace(0, len(signal) / sample_rate, len(signal), False)
    envelope = np.exp(-t / decay_time)
    return signal * envelope


def generate_sample(config: dict) -> np.ndarray:
    """Generate a sample based on configuration."""
    duration = config["duration"]

    if config.get("noise"):
        # Mix noise with some tone
        signal = generate_noise(duration) * 0.7
        signal += generate_sine_wave(config["freq"], duration) * 0.3
    else:
        signal = generate_sine_wave(config["freq"], duration)

    # Apply decay
    decay = config.get("decay", duration * 0.8)
    signal = apply_decay(signal, decay)

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.8

    return signal


def write_wav(filepath: Path, signal: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write signal to WAV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to 16-bit PCM
    signal_int = (signal * 32767).astype(np.int16)

    with wave.open(str(filepath), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int.tobytes())


def main():
    """Generate all placeholder samples."""
    base_dir = Path(__file__).parent.parent / "samples"

    print(f"Generating samples in {base_dir}")

    for genre in GENRES:
        for instrument, config in INSTRUMENTS.items():
            # Vary the config slightly per genre for some variety
            genre_config = config.copy()

            # Add some variation based on genre
            if genre == "lo-fi":
                genre_config["freq"] *= 0.9  # Lower pitch
            elif genre == "electronic":
                genre_config["freq"] *= 1.1  # Higher pitch
            elif genre == "jazz":
                genre_config["duration"] *= 1.2  # Longer notes

            signal = generate_sample(genre_config)

            output_path = base_dir / genre / instrument / f"{instrument}_01.wav"
            write_wav(output_path, signal)

            print(f"  Created: {output_path.relative_to(base_dir)}")

    print(f"\nDone! Created {len(GENRES) * len(INSTRUMENTS)} sample files.")


if __name__ == "__main__":
    main()
