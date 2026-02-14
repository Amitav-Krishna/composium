"""
Audio Visualizer Service

Generates visualization data for audio analysis debugging:
- Waveform (downsampled for display)
- Onset/beat detection markers
- Pitch contour
- Tempo analysis
"""

import io
import numpy as np
import librosa
import soundfile as sf
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
logger = logging.getLogger(__name__)


@dataclass
class VisualizationData:
    """Container for all visualization data."""
    # Waveform (downsampled)
    waveform: list[float]
    waveform_times: list[float]

    # Audio stats
    duration_seconds: float
    sample_rate: int

    # Onset detection
    onset_times: list[float]
    onset_strengths: list[float]

    # Beat tracking
    beat_times: list[float]
    tempo_bpm: float

    # Onset envelope (for visualization)
    onset_envelope: list[float]
    onset_envelope_times: list[float]

    # Pitch contour (if melodic)
    pitch_times: Optional[list[float]] = None
    pitch_frequencies: Optional[list[float]] = None
    pitch_confidences: Optional[list[float]] = None

    # RMS energy over time
    rms_values: list[float] = None
    rms_times: list[float] = None


def _load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio from bytes into numpy array."""
    # Try soundfile first
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y.astype(np.float32), sr
    except Exception:
        pass

    # Try librosa
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        return y, sr
    except Exception:
        pass

    # Fall back to pydub
    from pydub import AudioSegment as PydubSegment
    audio = PydubSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    y, sr = sf.read(wav_buffer)
    return y.astype(np.float32), sr


async def analyze_for_visualization(
    audio_bytes: bytes,
    downsample_factor: int = 100,
    include_pitch: bool = True,
) -> VisualizationData:
    """
    Analyze audio and return all visualization data.

    Args:
        audio_bytes: Raw audio file bytes
        downsample_factor: Factor to downsample waveform for display
        include_pitch: Whether to run pitch analysis (slower)

    Returns:
        VisualizationData with all analysis results
    """
    logger.info("=" * 60)
    logger.info("VISUALIZER: Starting audio analysis for visualization")

    # Load audio
    y, sr = _load_audio_from_bytes(audio_bytes)
    duration = len(y) / sr
    logger.info(f"VISUALIZER: Loaded audio: {len(y)} samples, sr={sr}, duration={duration:.2f}s")

    # Downsample waveform for display
    # Take every Nth sample
    waveform_indices = np.arange(0, len(y), downsample_factor)
    waveform = y[waveform_indices].tolist()
    waveform_times = (waveform_indices / sr).tolist()
    logger.info(f"VISUALIZER: Downsampled waveform to {len(waveform)} points")

    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env_times = librosa.times_like(onset_env, sr=sr)
    logger.info(f"VISUALIZER: Computed onset envelope: {len(onset_env)} frames")

    # Normalize onset envelope
    if np.max(onset_env) > 0:
        onset_env_normalized = onset_env / np.max(onset_env)
    else:
        onset_env_normalized = onset_env

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else []
    if len(onset_strengths) > 0 and np.max(onset_strengths) > 0:
        onset_strengths = onset_strengths / np.max(onset_strengths)
    logger.info(f"VISUALIZER: Detected {len(onset_times)} onsets")
    if len(onset_times) > 0:
        logger.info(f"VISUALIZER: Onset times: {[f'{t:.3f}s' for t in onset_times[:10]]}{'...' if len(onset_times) > 10 else ''}")

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Handle array return from newer librosa
    if hasattr(tempo, '__len__'):
        tempo_bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo_bpm = float(tempo) if tempo > 0 else 120.0

    logger.info(f"VISUALIZER: Detected tempo: {tempo_bpm:.1f} BPM")
    logger.info(f"VISUALIZER: Detected {len(beat_times)} beats")
    if len(beat_times) > 0:
        logger.info(f"VISUALIZER: Beat times: {[f'{t:.3f}s' for t in beat_times[:10]]}{'...' if len(beat_times) > 10 else ''}")

    # Compute RMS energy
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    logger.info(f"VISUALIZER: Computed RMS energy: {len(rms)} frames")

    # Pitch analysis (optional, slower)
    pitch_times_out = None
    pitch_frequencies = None
    pitch_confidences = None

    if include_pitch:
        logger.info("VISUALIZER: Running pitch analysis...")
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=80,
                fmax=800,
                sr=sr,
            )

            pitch_frame_times = librosa.times_like(f0, sr=sr)

            # Convert to lists, replacing NaN with None for JSON
            pitch_times_out = pitch_frame_times.tolist()
            pitch_frequencies = [float(f) if not np.isnan(f) else None for f in f0]
            pitch_confidences = voiced_probs.tolist() if voiced_probs is not None else None

            # Count voiced frames
            voiced_count = np.sum(~np.isnan(f0))
            logger.info(f"VISUALIZER: Pitch analysis: {voiced_count}/{len(f0)} voiced frames")
        except Exception as e:
            logger.warning(f"VISUALIZER: Pitch analysis failed: {e}")

    logger.info("VISUALIZER: Analysis complete")
    logger.info("=" * 60)

    return VisualizationData(
        waveform=waveform,
        waveform_times=waveform_times,
        duration_seconds=duration,
        sample_rate=sr,
        onset_times=onset_times.tolist(),
        onset_strengths=onset_strengths.tolist() if isinstance(onset_strengths, np.ndarray) else list(onset_strengths),
        beat_times=beat_times.tolist(),
        tempo_bpm=tempo_bpm,
        onset_envelope=onset_env_normalized.tolist(),
        onset_envelope_times=onset_env_times.tolist(),
        pitch_times=pitch_times_out,
        pitch_frequencies=pitch_frequencies,
        pitch_confidences=pitch_confidences,
        rms_values=rms.tolist(),
        rms_times=rms_times.tolist(),
    )
