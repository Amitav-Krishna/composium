"""
Audio Segmentation Service

The key innovation: Use Pulse STT word timestamps to identify speech regions.
Everything between speech regions is assumed to be musical content.
"""

import io
import uuid
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

from config.settings import settings
from app.models.schemas import AudioSegment, SegmentType
from app.services import transcription


class SpeechRegion:
    """Helper class for a detected speech region."""
    def __init__(self, start: float, end: float, text: str = ""):
        self.start = start
        self.end = end
        self.text = text


async def segment_recording(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    gap_threshold: float = 0.5,
    min_music_duration: float = 0.3,
    silence_threshold: float = 0.01,
) -> list[AudioSegment]:
    """
    Segment a recording into speech and music regions.

    Strategy:
    1. Send full recording to Pulse STT (get word timestamps)
    2. Build speech regions from word timestamps (merge close words)
    3. Gaps between speech = music segments
    4. Validate music segments with energy detection (filter out silence)
    5. Classify music segments: melodic vs rhythmic

    Args:
        audio_bytes: Raw audio file bytes
        content_type: MIME type of the audio
        gap_threshold: Max gap (seconds) between words to merge into one speech region
        min_music_duration: Minimum duration (seconds) for a valid music segment
        silence_threshold: RMS threshold below which is considered silence

    Returns:
        List of AudioSegment objects, sorted by start time
    """
    logger.info("=" * 60)
    logger.info("SEGMENTER: Starting audio segmentation")
    logger.info(f"  Input: {len(audio_bytes)} bytes, content_type={content_type}")

    # Step 1: Transcribe with timestamps
    logger.info("SEGMENTER: Step 1 - Sending to STT for transcription...")
    stt_response = await transcription.transcribe_with_timestamps(
        audio_bytes, content_type, word_timestamps=True
    )

    words = stt_response.get("words", [])
    full_transcript = stt_response.get("transcription", "")
    logger.info(f"SEGMENTER: STT returned {len(words)} words")
    logger.info(f"SEGMENTER: Full transcript: '{full_transcript}'")
    if words:
        logger.info(f"SEGMENTER: Word timestamps: {words}")

    # Load audio for duration and analysis
    logger.info("SEGMENTER: Step 2 - Loading audio for analysis...")
    y, sr = _load_audio_from_bytes(audio_bytes)
    total_duration = len(y) / sr
    logger.info(f"SEGMENTER: Audio loaded: {len(y)} samples, sr={sr}, duration={total_duration:.2f}s")
    logger.info(f"SEGMENTER: Audio stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

    # Step 2: Build speech regions from word timestamps
    speech_regions = _merge_close_words(words, gap_threshold)
    logger.info(f"SEGMENTER: Found {len(speech_regions)} speech regions")

    # Create speech segments
    segments = []
    for i, region in enumerate(speech_regions):
        segment = AudioSegment(
            id=str(uuid.uuid4()),
            type=SegmentType.SPEECH,
            start_seconds=region.start,
            end_seconds=region.end,
            transcript=region.text,
        )
        segments.append(segment)
        logger.info(f"SEGMENTER: Speech segment {i+1}: {region.start:.2f}s - {region.end:.2f}s = '{region.text}'")

    # Step 3: Find gaps between speech (potential music segments)
    music_regions = _find_gaps(speech_regions, total_duration, min_music_duration)
    logger.info(f"SEGMENTER: Step 3 - Found {len(music_regions)} potential music regions (gaps between speech)")
    for i, region in enumerate(music_regions):
        logger.info(f"SEGMENTER: Music region {i+1}: {region.start:.2f}s - {region.end:.2f}s (duration: {region.end - region.start:.2f}s)")

    # Step 4: Filter out silence and classify music segments
    logger.info("SEGMENTER: Step 4 - Analyzing music regions (filtering silence, classifying)...")
    for region in music_regions:
        # Extract audio chunk
        start_sample = int(region.start * sr)
        end_sample = int(region.end * sr)
        chunk = y[start_sample:end_sample]

        # Check if it's silence
        rms = np.sqrt(np.mean(chunk ** 2))
        logger.info(f"SEGMENTER: Region {region.start:.2f}-{region.end:.2f}s: RMS={rms:.6f}, threshold={silence_threshold}")
        if rms < silence_threshold:
            logger.info(f"SEGMENTER: -> SKIPPED (silence)")
            continue  # Skip silent regions

        # Step 5: Classify as melody or rhythm
        segment_type = _classify_music_segment(chunk, sr)
        logger.info(f"SEGMENTER: -> Classified as: {segment_type.value}")

        segment = AudioSegment(
            id=str(uuid.uuid4()),
            type=segment_type,
            start_seconds=region.start,
            end_seconds=region.end,
        )
        segments.append(segment)

    # Sort by start time
    segments.sort(key=lambda s: s.start_seconds)

    logger.info(f"SEGMENTER: Final result: {len(segments)} segments")
    for seg in segments:
        logger.info(f"SEGMENTER:   - {seg.type.value}: {seg.start_seconds:.2f}s - {seg.end_seconds:.2f}s")
    logger.info("=" * 60)

    return segments


def _merge_close_words(
    words: list[dict],
    gap_threshold: float,
    max_word_duration: float = 1.5,
) -> list[SpeechRegion]:
    """Merge words that are close together into speech regions.

    Args:
        words: Word dicts with 'start', 'end', 'word' keys from STT.
        gap_threshold: Max gap (seconds) between words to merge.
        max_word_duration: Cap on a single word's duration. STT engines
            often stretch a word's end timestamp across musical content
            (beatboxing, humming) that follows. Capping prevents this
            from swallowing music regions into speech.
    """
    if not words:
        return []

    regions = []
    w0 = words[0]
    current_start = w0.get("start", 0)
    current_end = min(w0.get("end", 0), current_start + max_word_duration)
    current_text = w0.get("word", "")

    for word in words[1:]:
        word_start = word.get("start", 0)
        word_end = min(word.get("end", 0), word_start + max_word_duration)
        word_text = word.get("word", "")

        # If gap between current region and this word is small, merge
        if word_start - current_end <= gap_threshold:
            current_end = word_end
            current_text += " " + word_text
        else:
            # Save current region and start a new one
            regions.append(SpeechRegion(current_start, current_end, current_text.strip()))
            current_start = word_start
            current_end = word_end
            current_text = word_text

    # Don't forget the last region
    regions.append(SpeechRegion(current_start, current_end, current_text.strip()))

    return regions


def _find_gaps(
    speech_regions: list[SpeechRegion],
    total_duration: float,
    min_duration: float,
) -> list[SpeechRegion]:
    """Find gaps between speech regions (potential music segments)."""
    gaps = []

    if not speech_regions:
        # Entire recording is a gap (music)
        if total_duration >= min_duration:
            gaps.append(SpeechRegion(0, total_duration))
        return gaps

    # Gap before first speech region
    if speech_regions[0].start >= min_duration:
        gaps.append(SpeechRegion(0, speech_regions[0].start))

    # Gaps between speech regions
    for i in range(len(speech_regions) - 1):
        gap_start = speech_regions[i].end
        gap_end = speech_regions[i + 1].start
        if gap_end - gap_start >= min_duration:
            gaps.append(SpeechRegion(gap_start, gap_end))

    # Gap after last speech region
    if total_duration - speech_regions[-1].end >= min_duration:
        gaps.append(SpeechRegion(speech_regions[-1].end, total_duration))

    return gaps


def _classify_music_segment(audio_chunk: np.ndarray, sr: int) -> SegmentType:
    """
    Classify a music segment as melody (pitched) or rhythm (percussive).

    Strategy: Run pitch detection (librosa.pyin).
    If strong pitch confidence > threshold → melody
    If mostly unpitched / percussive → rhythm
    """
    try:
        # Use pyin for pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_chunk,
            fmin=80,
            fmax=800,
            sr=sr,
        )

        # Calculate ratio of voiced frames
        if voiced_flag is not None and len(voiced_flag) > 0:
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        else:
            voiced_ratio = 0

        # If more than 40% of frames have clear pitch, it's melodic
        if voiced_ratio > 0.4:
            return SegmentType.MELODY
        else:
            return SegmentType.RHYTHM

    except Exception:
        # Default to rhythm if pitch detection fails
        return SegmentType.RHYTHM


def _load_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load audio from bytes into numpy array."""
    logger.info(f"AUDIO_LOAD: Attempting to load {len(audio_bytes)} bytes of audio")

    # First, try direct loading with soundfile
    try:
        logger.info("AUDIO_LOAD: Trying soundfile...")
        y, sr = sf.read(io.BytesIO(audio_bytes))
        logger.info(f"AUDIO_LOAD: soundfile succeeded! shape={y.shape}, sr={sr}")
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            logger.info(f"AUDIO_LOAD: Converted stereo to mono")
        if sr != 22050:
            logger.info(f"AUDIO_LOAD: Resampling from {sr} to 22050")
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        return y.astype(np.float32), sr
    except Exception as e:
        logger.info(f"AUDIO_LOAD: soundfile failed: {e}")

    # Try librosa directly
    try:
        logger.info("AUDIO_LOAD: Trying librosa...")
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
        logger.info(f"AUDIO_LOAD: librosa succeeded! shape={y.shape}, sr={sr}")
        return y, sr
    except Exception as e:
        logger.info(f"AUDIO_LOAD: librosa failed: {e}")

    # Fall back to pydub for format conversion (handles webm, mp3, etc.)
    logger.info("AUDIO_LOAD: Trying pydub (ffmpeg) for format conversion...")
    from pydub import AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    logger.info(f"AUDIO_LOAD: pydub loaded: duration={len(audio)}ms, channels={audio.channels}, frame_rate={audio.frame_rate}")
    audio = audio.set_frame_rate(22050).set_channels(1)

    # Export to WAV and load with soundfile
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    y, sr = sf.read(wav_buffer)
    logger.info(f"AUDIO_LOAD: pydub conversion succeeded! shape={y.shape}, sr={sr}")
    return y.astype(np.float32), sr


async def extract_segment_audio(
    audio_bytes: bytes,
    segment: AudioSegment,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Extract audio for a specific segment and save to file.

    Returns:
        Path to the extracted audio file
    """
    out_dir = Path(output_dir or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = _load_audio_from_bytes(audio_bytes)

    start_sample = int(segment.start_seconds * sr)
    end_sample = int(segment.end_seconds * sr)
    chunk = y[start_sample:end_sample]

    filename = f"segment_{segment.id[:8]}.wav"
    output_path = out_dir / filename

    sf.write(str(output_path), chunk, sr)

    return str(output_path)
