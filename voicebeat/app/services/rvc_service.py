"""
RVC Voice Conversion Service

Converts TTS audio through a trained RVC model for more natural voice quality.
Optional — when no model path is configured, returns input audio unchanged.
"""

import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from config.settings import settings

logger = logging.getLogger(__name__)

_rvc_instance = None
_rvc_loaded = False


def _get_rvc():
    """Lazy singleton — loads RVC model on first call only."""
    global _rvc_instance, _rvc_loaded

    if _rvc_loaded:
        return _rvc_instance

    _rvc_loaded = True

    # Guard 1: model path configured?
    if not settings.rvc_model_path:
        logger.info("rvc_service: RVC disabled (RVC_MODEL_PATH not set)")
        return None

    # Guard 2: rvc-python importable?
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        logger.warning(
            "rvc_service: rvc-python not installed — RVC disabled. "
            "Install with: pip install rvc-python torch torchaudio"
        )
        return None

    # Guard 3: model file exists?
    model_path = Path(settings.rvc_model_path)
    if not model_path.exists():
        logger.warning(
            f"rvc_service: model file not found at {model_path} — RVC disabled"
        )
        return None

    # Auto-detect GPU
    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    f0_method = "rmvpe" if device.startswith("cuda") else "harvest"

    logger.info(
        f"rvc_service: loading model {model_path.name} "
        f"(device={device}, f0_method={f0_method})"
    )

    try:
        rvc = RVCInference(device=device)
        rvc.load_model(str(model_path))
        rvc.set_params(
            f0method=f0_method,
            filter_radius=3,      # median filter for smoother pitch
            resample_sr=0,        # no extra resampling
            rms_mix_rate=0.25,    # blend RVC volume envelope with original
            protect=0.33,         # protect voiceless consonants
        )

        # Optional index file for better voice matching
        if settings.rvc_index_path:
            index_path = Path(settings.rvc_index_path)
            if index_path.exists():
                rvc.set_params(index_path=str(index_path))
                logger.info(f"rvc_service: loaded index file {index_path.name}")
            else:
                logger.warning(
                    f"rvc_service: index file not found at {index_path}, skipping"
                )

        _rvc_instance = rvc
        logger.info("rvc_service: model loaded successfully")
        return rvc
    except Exception as e:
        logger.warning(f"rvc_service: failed to load model: {e}")
        return None


_MIN_DURATION_SEC = 3.0  # RVC needs ~3s of audio for good quality


def _pad_audio(audio_bytes: bytes) -> tuple[bytes, int]:
    """Loop short audio to meet minimum duration for RVC quality.

    Returns (padded_wav_bytes, original_sample_count).
    """
    y, sr = sf.read(io.BytesIO(audio_bytes))
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    original_len = len(y)
    min_samples = int(_MIN_DURATION_SEC * sr)

    if original_len >= min_samples:
        return audio_bytes, original_len

    # Loop the audio until we reach minimum duration
    repeats = int(np.ceil(min_samples / original_len))
    padded = np.tile(y, repeats)[:min_samples]

    # Smooth the loop joints to avoid clicks
    fade = min(256, original_len // 4)
    if fade > 1:
        for joint in range(1, repeats):
            idx = joint * original_len
            if idx + fade < len(padded) and idx - fade >= 0:
                ramp = np.linspace(0, 1, fade)
                padded[idx - fade:idx] *= 1 - ramp
                padded[idx:idx + fade] *= ramp

    buf = io.BytesIO()
    sf.write(buf, padded, sr, format="WAV")
    buf.seek(0)

    logger.info(
        f"rvc_service: padded {original_len / sr:.1f}s -> "
        f"{len(padded) / sr:.1f}s for better RVC quality"
    )
    return buf.read(), original_len


def _trim_output(output_bytes: bytes, original_sample_count: int) -> bytes:
    """Trim RVC output back to original duration."""
    y, sr = sf.read(io.BytesIO(output_bytes))
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    if len(y) > original_sample_count:
        y = y[:original_sample_count]

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _convert_sync(audio_bytes: bytes, pitch_semitones: int) -> bytes:
    """Synchronous RVC inference via temp files, with audio padding."""
    rvc = _get_rvc()
    if rvc is None:
        return audio_bytes

    # Pad short audio so RVC has enough context
    padded_bytes, original_len = _pad_audio(audio_bytes)

    input_fd, input_path = tempfile.mkstemp(suffix=".wav")
    output_fd, output_path = tempfile.mkstemp(suffix=".wav")

    try:
        os.close(input_fd)
        os.close(output_fd)

        with open(input_path, "wb") as f:
            f.write(padded_bytes)

        if pitch_semitones != 0:
            rvc.set_params(f0up_key=pitch_semitones)

        rvc.infer_file(input_path, output_path)

        with open(output_path, "rb") as f:
            result = f.read()

        # Trim back to original duration if we padded
        result = _trim_output(result, original_len)

        return result
    except Exception as e:
        logger.warning(f"rvc_service: inference failed, returning original audio: {e}")
        return audio_bytes
    finally:
        for p in (input_path, output_path):
            try:
                os.unlink(p)
            except OSError:
                pass


async def convert_audio(audio_bytes: bytes, pitch_semitones: int = 0) -> bytes:
    """
    Convert audio through RVC model.

    Returns input unchanged if RVC is not configured or any error occurs.

    Args:
        audio_bytes: WAV audio bytes to convert.
        pitch_semitones: Pitch shift in semitones (0 = no shift).

    Returns:
        Converted WAV audio bytes, or original bytes on failure/disabled.
    """
    if not settings.rvc_model_path:
        return audio_bytes

    return await asyncio.to_thread(_convert_sync, audio_bytes, pitch_semitones)
