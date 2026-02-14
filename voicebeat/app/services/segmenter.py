"""
Audio Segmentation Service

Hybrid approach:
1. STT transcription gives the full transcript + word timestamps.
2. A lightweight LLM call finds where the semantic command ends and
   beat vocalizations begin (e.g. "please generate me a guitar | bum bum bum").
3. The command portion is parsed for instrument/genre via description_parser.
4. The beat portion is analysed purely acoustically using librosa onset
   detection — no reliance on transcribed beat words.
"""

import io
import json
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

from openai import AsyncOpenAI
from config.settings import settings
from app.models.schemas import AudioSegment, Instrument, SegmentType
from app.services import transcription


_openai_client: Optional[AsyncOpenAI] = None

def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ---------------------------------------------------------------------------
# Prompt for boundary detection
# ---------------------------------------------------------------------------

_BOUNDARY_PROMPT = """\
You are analyzing a voice recording where a user gives a music instruction \
followed by a vocal demonstration (beatboxing, humming, or syllabic sounds).

Identify where the semantic command ends and the vocal demonstration begins.

Vocal / beat words are short syllabic sounds like: bum, da, boom, tss, ka, \
dun, ba, na, la, hmm, dum, psh, ch, etc.
Command words are natural language: "generate", "make", "I want", "that goes", \
"give me", "please", "create", instrument names used as nouns, etc.

Return ONLY valid JSON:
{{"command_end_seconds": <float>, "command": "<command text only>"}}

Rules:
- "command_end_seconds" is the end timestamp (seconds) of the last command word.
- If there are NO vocal demonstrations (pure command), set command_end_seconds \
to the total duration.
- If there is NO command at all (pure beats), set command_end_seconds to 0.0 \
and "command" to "".

Transcript: {transcript}
Word timestamps (JSON): {words}
Total duration: {duration:.2f}s
"""


class SpeechRegion:
    """Helper class for a detected speech region."""
    def __init__(self, start: float, end: float, text: str = ""):
        self.start = start
        self.end = end
        self.text = text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def segment_recording(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    min_beat_duration: float = 0.2,
    silence_threshold: float = 0.01,
) -> list[AudioSegment]:
    """
    Segment a recording into a (optional) SPEECH segment and a (optional)
    MELODY/RHYTHM segment, enriched with instrument metadata.

    Strategy:
    1. STT with word timestamps on the full recording.
    2. LLM boundary call → split point between command and beat vocalizations.
    3. Instrument extracted from command text via description_parser.
    4. Beat region classified acoustically (pyin voiced-ratio → MELODY/RHYTHM).

    Returns:
        List of AudioSegment objects sorted by start time.
    """
    logger.info("=" * 60)
    logger.info("SEGMENTER: Starting hybrid segmentation")
    logger.info(f"  Input: {len(audio_bytes)} bytes, content_type={content_type}")

    # Step 1: STT with word timestamps
    logger.info("SEGMENTER: Step 1 - Transcribing with word timestamps...")
    stt_response = await transcription.transcribe_with_timestamps(
        audio_bytes, content_type, word_timestamps=True
    )
    words = stt_response.get("words", [])
    full_transcript = stt_response.get("transcription", "")
    logger.info(f"SEGMENTER: Transcript: '{full_transcript}'")
    logger.info(f"SEGMENTER: {len(words)} word timestamps returned")

    # Step 2: Load audio
    logger.info("SEGMENTER: Step 2 - Loading audio...")
    y, sr = _load_audio_from_bytes(audio_bytes)
    total_duration = len(y) / sr
    logger.info(f"SEGMENTER: duration={total_duration:.2f}s, sr={sr}")

    # Step 3: LLM boundary detection
    logger.info("SEGMENTER: Step 3 - Finding command/beat boundary via LLM...")
    boundary_seconds, command_text = await _find_command_boundary(
        words, full_transcript, total_duration
    )
    logger.info(f"SEGMENTER: Boundary at {boundary_seconds:.2f}s | command='{command_text}'")

    segments: list[AudioSegment] = []

    # Step 4: Extract instrument from the command (before building segments so
    # both the speech and beat segments can carry it)
    instrument: Optional[Instrument] = None
    if command_text.strip():
        instrument = await _extract_instrument(command_text)
        logger.info(f"SEGMENTER: Detected instrument: {instrument}")

    # Step 5: Build SPEECH segment if there is a command
    if command_text.strip() and boundary_seconds > 0:
        speech_chunk = y[: int(boundary_seconds * sr)]
        speech_volume = _rms_to_volume(speech_chunk)
        speech_seg = AudioSegment(
            id=str(uuid.uuid4()),
            type=SegmentType.SPEECH,
            start_seconds=0.0,
            end_seconds=boundary_seconds,
            transcript=command_text,
            semantic_command=command_text,
            instrument=instrument,
            volume=speech_volume,
        )
        segments.append(speech_seg)
        logger.info(f"SEGMENTER: Speech segment 0.0s-{boundary_seconds:.2f}s: '{command_text}' instrument={instrument} volume={speech_volume:.3f}")

    # Step 6: Beat region = everything after the command
    beat_start = boundary_seconds
    beat_end = total_duration

    if beat_end - beat_start >= min_beat_duration:
        start_sample = int(beat_start * sr)
        end_sample = int(beat_end * sr)
        beat_chunk = y[start_sample:end_sample]

        rms = float(np.sqrt(np.mean(beat_chunk ** 2)))
        logger.info(f"SEGMENTER: Beat region {beat_start:.2f}s-{beat_end:.2f}s, RMS={rms:.6f}")

        if rms >= silence_threshold:
            seg_type = _classify_music_segment(beat_chunk, sr)
            beat_volume = _rms_to_volume(beat_chunk)
            logger.info(f"SEGMENTER: Beat classified as {seg_type.value}, volume={beat_volume:.3f}")

            beat_seg = AudioSegment(
                id=str(uuid.uuid4()),
                type=seg_type,
                start_seconds=beat_start,
                end_seconds=beat_end,
                instrument=instrument,
                semantic_command=command_text if command_text.strip() else None,
                volume=beat_volume,
            )
            segments.append(beat_seg)
        else:
            logger.info("SEGMENTER: Beat region is silent, skipping")
    else:
        logger.info(f"SEGMENTER: Beat region too short ({beat_end - beat_start:.2f}s), skipping")

    segments.sort(key=lambda s: s.start_seconds)

    logger.info(f"SEGMENTER: Done — {len(segments)} segment(s)")
    for seg in segments:
        logger.info(
            f"  {seg.type.value}: {seg.start_seconds:.2f}s-{seg.end_seconds:.2f}s"
            f"  instrument={seg.instrument}  cmd='{seg.semantic_command}'"
        )
    logger.info("=" * 60)
    return segments


# ---------------------------------------------------------------------------
# LLM boundary detection
# ---------------------------------------------------------------------------

async def _find_command_boundary(
    words: list[dict],
    transcript: str,
    total_duration: float,
) -> tuple[float, str]:
    """
    Ask GPT-4o-mini to split the transcript into semantic command vs beat vocalizations.

    Returns:
        (command_end_seconds, command_text)
    """
    if not transcript.strip():
        return 0.0, ""

    # If there are no word timestamps, fall back to treating entire audio as beats
    if not words:
        return 0.0, transcript

    prompt = _BOUNDARY_PROMPT.format(
        transcript=transcript,
        words=json.dumps(words),
        duration=total_duration,
    )

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        command_end = float(data.get("command_end_seconds", total_duration))
        command_text = str(data.get("command", transcript)).strip()
        # Clamp to valid range
        command_end = max(0.0, min(command_end, total_duration))
        return command_end, command_text
    except Exception as e:
        logger.warning(f"SEGMENTER: LLM boundary detection failed ({e}), treating full audio as beats")
        return 0.0, transcript


# ---------------------------------------------------------------------------
# Instrument extraction
# ---------------------------------------------------------------------------

async def _extract_instrument(command_text: str) -> Optional[Instrument]:
    """
    Extract the requested instrument from a command string.
    Uses description_parser so we reuse the existing logic.
    """
    from app.services import description_parser
    try:
        desc = await description_parser.parse_description(command_text)
        if desc.instruments:
            return desc.instruments[0]
    except Exception as e:
        logger.warning(f"SEGMENTER: Instrument extraction failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Acoustic helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _rms_to_volume(audio_chunk: np.ndarray) -> float:
    """
    Convert an audio chunk to a normalised volume value (0.0–1.0).

    Uses RMS energy mapped through a reference ceiling of 0.2 RMS, which
    corresponds to a loud but undistorted vocal recording. Values above that
    are clamped to 1.0.
    """
    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
    return min(rms / 0.2, 1.0)


def _classify_music_segment(audio_chunk: np.ndarray, sr: int) -> SegmentType:
    """
    Classify a music segment as melody (pitched) or rhythm (percussive).

    Uses librosa.pyin pitch detection.
    voiced_ratio > 0.4 → MELODY, otherwise → RHYTHM.
    """
    try:
        voiced_flag = librosa.pyin(audio_chunk, fmin=80, fmax=800, sr=sr)[1]
        if voiced_flag is not None and len(voiced_flag) > 0:
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        else:
            voiced_ratio = 0.0

        return SegmentType.MELODY if voiced_ratio > 0.4 else SegmentType.RHYTHM
    except Exception:
        return SegmentType.RHYTHM


def _load_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load audio bytes into a mono float32 numpy array at 22050 Hz."""
    logger.info(f"AUDIO_LOAD: Loading {len(audio_bytes)} bytes")

    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        return y.astype(np.float32), sr
    except Exception as e:
        logger.info(f"AUDIO_LOAD: soundfile failed: {e}")

    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
        return y, sr
    except Exception as e:
        logger.info(f"AUDIO_LOAD: librosa failed: {e}")

    from pydub import AudioSegment as PydubSegment
    audio = PydubSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(22050).set_channels(1)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    y, sr = sf.read(wav_buffer)
    return y.astype(np.float32), sr


async def extract_segment_audio(
    audio_bytes: bytes,
    segment: AudioSegment,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Extract the audio slice for a segment and save to a WAV file.

    Returns:
        Path to the extracted audio file.
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
