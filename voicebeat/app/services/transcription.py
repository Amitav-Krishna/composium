import httpx
import logging
from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings

logger = logging.getLogger(__name__)

PULSE_STT_URL = "https://waves-api.smallest.ai/api/v1/pulse/get_text"


async def transcribe(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    language: str = "en",
    word_timestamps: bool = False,
) -> str:
    """
    Transcribe audio bytes to text using Smallest.ai Pulse STT.

    Args:
        audio_bytes: Raw audio file bytes
        content_type: MIME type of the audio (audio/wav, audio/mpeg, etc.)
        language: Language code (default: "en")
        word_timestamps: Whether to include word-level timestamps

    Returns:
        Transcribed text string
    """
    result = await transcribe_with_timestamps(
        audio_bytes, content_type, language, word_timestamps
    )
    return result.get("transcription", "")


async def transcribe_with_timestamps(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    language: str = "en",
    word_timestamps: bool = True,
) -> dict:
    """
    Transcribe audio bytes to text with full response including timestamps.

    Args:
        audio_bytes: Raw audio file bytes
        content_type: MIME type of the audio
        language: Language code
        word_timestamps: Whether to include word-level timestamps

    Returns:
        Full API response dict with transcription, words, and utterances
    """
    logger.info("=" * 60)
    logger.info("STT: Sending audio to Pulse STT API")
    logger.info(f"STT: Audio size: {len(audio_bytes)} bytes")
    logger.info(f"STT: Content-Type: {content_type}")
    logger.info(f"STT: Model: {settings.pulse_model}")

    params = {
        "model": settings.pulse_model,
        "language": language,
        "word_timestamps": str(word_timestamps).lower(),
    }

    headers = {
        "Authorization": f"Bearer {settings.smallest_api_key}",
        "Content-Type": content_type,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            PULSE_STT_URL,
            params=params,
            headers=headers,
            content=audio_bytes,
        )
        logger.info(f"STT: Response status: {response.status_code}")
        response.raise_for_status()
        result = response.json()

        logger.info(f"STT: Response keys: {list(result.keys())}")
        logger.info(f"STT: Transcription: '{result.get('transcription', '')}'")
        logger.info(f"STT: Words count: {len(result.get('words', []))}")
        if result.get('words'):
            logger.info(f"STT: Words detail: {result.get('words')}")
        logger.info("=" * 60)

        return result


def get_content_type_for_extension(filename: str) -> str:
    """Map file extension to content type."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mapping = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "webm": "audio/webm",
    }
    return mapping.get(ext, "audio/wav")
