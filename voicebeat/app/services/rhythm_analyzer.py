import io
import librosa
import numpy as np
import soundfile as sf
from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings
from app.models.schemas import RhythmPattern, QuantizedBeat, Instrument


async def analyze_rhythm(
    audio_bytes: bytes,
    target_bpm: Optional[int] = None,
    subdivisions: int = 16,
) -> tuple[RhythmPattern, list[float]]:
    """
    Analyze rhythm from audio using onset detection and beat tracking.

    Args:
        audio_bytes: Raw audio file bytes
        target_bpm: Target BPM to quantize to (auto-detected if None)
        subdivisions: Number of subdivisions per bar (default: 16 for 16th notes)

    Returns:
        Tuple of (RhythmPattern, list of raw onset times in seconds)
    """
    # Load audio from bytes
    y, sr = _load_audio_from_bytes(audio_bytes)

    # Detect BPM if not provided
    if target_bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Handle array return from newer librosa versions
        if hasattr(tempo, '__len__'):
            bpm = int(tempo[0]) if len(tempo) > 0 else settings.default_bpm
        else:
            bpm = int(tempo) if tempo > 0 else settings.default_bpm
    else:
        bpm = target_bpm

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Get onset strengths for velocity mapping
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else []

    # Normalize strengths to 0-1
    if len(onset_strengths) > 0:
        max_strength = np.max(onset_strengths)
        if max_strength > 0:
            onset_strengths = onset_strengths / max_strength
        else:
            onset_strengths = np.ones_like(onset_strengths)

    # Calculate timing
    seconds_per_beat = 60.0 / bpm
    seconds_per_subdivision = seconds_per_beat / (subdivisions / 4)  # 4 beats per bar

    # Quantize onsets to grid
    beats = []
    for i, onset_time in enumerate(onset_times):
        # Calculate position in subdivisions from start
        total_subdivisions = onset_time / seconds_per_subdivision

        # Round to nearest subdivision
        quantized_pos = int(round(total_subdivisions))

        # Calculate bar and position within bar
        bar = quantized_pos // subdivisions
        position = quantized_pos % subdivisions

        # Get velocity
        velocity = float(onset_strengths[i]) if i < len(onset_strengths) else 1.0

        beats.append(QuantizedBeat(
            position=position,
            bar=bar,
            instrument=Instrument.KICK,  # Default, will be assigned later
            velocity=velocity,
        ))

    # Calculate number of bars
    if len(beats) > 0:
        bars = max(b.bar for b in beats) + 1
    else:
        bars = 1

    pattern = RhythmPattern(
        beats=beats,
        bpm=bpm,
        bars=bars,
        subdivisions=subdivisions,
        time_signature="4/4",
    )

    return pattern, onset_times.tolist()


def _load_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load audio from bytes into numpy array."""
    # Try to load with soundfile first (handles WAV, FLAC, OGG)
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        # Resample to 22050 for librosa compatibility
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        return y.astype(np.float32), sr
    except Exception:
        pass

    # Fallback to librosa for MP3 and other formats
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
    return y, sr


def assign_instruments(
    pattern: RhythmPattern,
    available_instruments: list[Instrument],
) -> RhythmPattern:
    """
    Assign instruments to beats based on position heuristics.

    Heuristics:
    - Positions 0, 8 (downbeats) -> kick
    - Positions 4, 12 (backbeats) -> snare
    - Off-beats -> hi-hat
    - Other positions -> other available instruments
    """
    kick = Instrument.KICK if Instrument.KICK in available_instruments else None
    snare = Instrument.SNARE if Instrument.SNARE in available_instruments else None
    hihat = Instrument.HI_HAT if Instrument.HI_HAT in available_instruments else None

    # Get other instruments for variety
    others = [i for i in available_instruments if i not in [Instrument.KICK, Instrument.SNARE, Instrument.HI_HAT]]

    new_beats = []
    for beat in pattern.beats:
        pos = beat.position

        # Assign instrument based on position
        if pos in [0, 8] and kick:
            instrument = kick
        elif pos in [4, 12] and snare:
            instrument = snare
        elif pos % 2 == 1 and hihat:  # Off-beats
            instrument = hihat
        elif hihat:
            instrument = hihat
        elif others:
            instrument = others[pos % len(others)]
        else:
            instrument = available_instruments[0] if available_instruments else Instrument.KICK

        new_beats.append(QuantizedBeat(
            position=beat.position,
            bar=beat.bar,
            instrument=instrument,
            velocity=beat.velocity,
        ))

    return RhythmPattern(
        beats=new_beats,
        bpm=pattern.bpm,
        bars=pattern.bars,
        subdivisions=pattern.subdivisions,
        time_signature=pattern.time_signature,
    )
