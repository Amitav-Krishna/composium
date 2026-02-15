import asyncio
import logging
import random
import sys
from pathlib import Path

import httpx
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings

logger = logging.getLogger(__name__)

WAVES_TTS_URL = "https://waves-api.smallest.ai/api/v1/lightning-v2/get_speech"

_word_cache: dict[tuple[str, str], bytes] = {}

# Limit concurrent TTS requests to avoid 429s
_semaphore = asyncio.Semaphore(2)

MAX_RETRIES = 5


async def speak_word(word: str, voice_id: str | None = None) -> bytes:
    """Cached single-word TTS. Avoids redundant API calls for repeated words."""
    # Strip syllable fragment markers (e.g. "Mu-" -> "Mu", "-sic" -> "sic")
    clean = word.strip().strip("-")
    if not clean:
        clean = word.strip()
    cache_key = (clean.lower(), voice_id or settings.tts_voice_id)
    if cache_key in _word_cache:
        return _word_cache[cache_key]
    audio = await speak(clean, voice_id)
    _word_cache[cache_key] = audio
    return audio


async def speak(
    text: str,
    voice_id: str | None = None,
    sample_rate: int | None = None,
) -> bytes:
    """
    Convert text to speech using Smallest.ai Waves Lightning v2 TTS.
    Retries with exponential backoff on 429 rate-limit errors.
    """
    voice = voice_id or settings.tts_voice_id
    rate = sample_rate or settings.tts_sample_rate

    # Parse comma-separated API keys and select one randomly
    api_keys = [key.strip() for key in settings.smallest_api_key.split(",")]

    headers = {
        "Authorization": f"Bearer {random.choice(api_keys)}",
        "Content-Type": "application/json",
    }

    payload = {
        "voice_id": voice,
        "text": text,
        "sample_rate": rate,
        "output_format": "wav",
    }

    async with _semaphore:
        for attempt in range(MAX_RETRIES):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    WAVES_TTS_URL,
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 200:
                return response.content

            if response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"TTS 429 for '{text}', retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
                continue

            # Non-retryable error
            logger.error(f"TTS API error {response.status_code}: {response.text} | payload={payload}")
            response.raise_for_status()

        # All retries exhausted
        logger.error(f"TTS failed after {MAX_RETRIES} retries for '{text}'")
        response.raise_for_status()


async def speak_to_file(
    text: str,
    output_path: str | Path,
    voice_id: str | None = None,
    sample_rate: int | None = None,
) -> Path:
    """Convert text to speech and save to a file."""
    audio_bytes = await speak(text, voice_id, sample_rate)
    output = Path(output_path)
    output.write_bytes(audio_bytes)
    return output
