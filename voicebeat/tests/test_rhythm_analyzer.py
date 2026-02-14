import pytest
import numpy as np
import wave
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import RhythmPattern, QuantizedBeat, Instrument
from app.services.rhythm_analyzer import _load_audio_from_bytes, assign_instruments


def generate_test_wav(duration_sec: float = 1.0, sample_rate: int = 22050) -> bytes:
    """Generate a simple test WAV file with clicks at regular intervals."""
    # Create audio with clicks at quarter note intervals
    num_samples = int(sample_rate * duration_sec)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Add clicks every quarter second (4 clicks for 1 second at 120 BPM)
    click_interval = sample_rate // 4
    click_duration = int(sample_rate * 0.01)  # 10ms clicks

    for i in range(4):
        start = i * click_interval
        end = min(start + click_duration, num_samples)
        # Simple click: sharp attack, quick decay
        click = np.linspace(1, 0, end - start)
        audio[start:end] = click

    # Convert to 16-bit PCM
    audio_int = (audio * 32767).astype(np.int16)

    # Write to WAV
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())

    return buffer.getvalue()


class TestLoadAudioFromBytes:
    """Tests for loading audio from bytes."""

    def test_load_wav_file(self):
        """Test loading a WAV file from bytes."""
        wav_bytes = generate_test_wav()
        y, sr = _load_audio_from_bytes(wav_bytes)

        assert isinstance(y, np.ndarray)
        assert sr == 22050
        assert len(y) > 0

    def test_loaded_audio_is_mono(self):
        """Test that loaded audio is mono."""
        wav_bytes = generate_test_wav()
        y, sr = _load_audio_from_bytes(wav_bytes)

        # Should be 1D array (mono)
        assert len(y.shape) == 1


class TestAssignInstruments:
    """Tests for instrument assignment heuristics."""

    def test_assign_kick_on_downbeats(self):
        """Test that kicks are assigned to positions 0 and 8."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=0, bar=0),
                QuantizedBeat(position=8, bar=0),
            ],
            bpm=120,
            bars=1,
        )

        instruments = [Instrument.KICK, Instrument.SNARE, Instrument.HI_HAT]
        result = assign_instruments(pattern, instruments)

        assert result.beats[0].instrument == Instrument.KICK
        assert result.beats[1].instrument == Instrument.KICK

    def test_assign_snare_on_backbeats(self):
        """Test that snares are assigned to positions 4 and 12."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=4, bar=0),
                QuantizedBeat(position=12, bar=0),
            ],
            bpm=120,
            bars=1,
        )

        instruments = [Instrument.KICK, Instrument.SNARE, Instrument.HI_HAT]
        result = assign_instruments(pattern, instruments)

        assert result.beats[0].instrument == Instrument.SNARE
        assert result.beats[1].instrument == Instrument.SNARE

    def test_assign_hihat_on_offbeats(self):
        """Test that hi-hats are assigned to odd positions (off-beats)."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=1, bar=0),
                QuantizedBeat(position=3, bar=0),
                QuantizedBeat(position=5, bar=0),
            ],
            bpm=120,
            bars=1,
        )

        instruments = [Instrument.KICK, Instrument.SNARE, Instrument.HI_HAT]
        result = assign_instruments(pattern, instruments)

        for beat in result.beats:
            assert beat.instrument == Instrument.HI_HAT

    def test_fallback_when_instrument_not_available(self):
        """Test fallback when preferred instrument not available."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=0, bar=0),  # Would be kick
            ],
            bpm=120,
            bars=1,
        )

        # Only piano available
        instruments = [Instrument.PIANO]
        result = assign_instruments(pattern, instruments)

        # Should fall back to available instrument
        assert result.beats[0].instrument == Instrument.PIANO

    def test_preserves_velocity_and_bar(self):
        """Test that velocity and bar info are preserved."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=0, bar=2, velocity=0.8),
            ],
            bpm=120,
            bars=3,
        )

        instruments = [Instrument.KICK]
        result = assign_instruments(pattern, instruments)

        assert result.beats[0].bar == 2
        assert result.beats[0].velocity == 0.8
