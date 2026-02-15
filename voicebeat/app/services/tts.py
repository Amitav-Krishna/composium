import asyncio
import io
import logging
import sys
import wave
from pathlib import Path

import httpx
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings

logger = logging.getLogger(__name__)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

_word_cache: dict[tuple[str, str], bytes] = {}
_phrase_cache: dict[tuple[str, str], bytes] = {}

# Limit concurrent TTS requests to avoid 429s
_semaphore = asyncio.Semaphore(2)

MAX_RETRIES = 5


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


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


async def speak_phrase(words: list[str], voice_id: str | None = None) -> bytes:
    """Synthesize a full sentence from a list of words for natural-sounding TTS."""
    sentence = " ".join(w.strip().strip("-") for w in words).strip()
    if not sentence:
        sentence = " ".join(words)
    vid = voice_id or settings.tts_voice_id
    cache_key = (sentence.lower(), vid)
    if cache_key in _phrase_cache:
        return _phrase_cache[cache_key]
    audio = await speak(sentence, voice_id)
    _phrase_cache[cache_key] = audio
    return audio


async def speak(
    text: str,
    voice_id: str | None = None,
    sample_rate: int | None = None,
) -> bytes:
    """
    Convert text to speech using ElevenLabs TTS API.
    Returns WAV bytes. Retries with exponential backoff on 429 rate-limit errors.
    """
    voice = voice_id or settings.tts_voice_id
    rate = sample_rate or settings.tts_sample_rate

    headers = {
        "xi-api-key": settings.elevenlabs_api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    # Request PCM output at the configured sample rate
    url = f"{ELEVENLABS_TTS_URL}/{voice}?output_format=pcm_{rate}"

    async with _semaphore:
        for attempt in range(MAX_RETRIES):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 200:
                wav_bytes = _pcm_to_wav(response.content, rate)
                return wav_bytes

            if response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"TTS 429 for '{text}', retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
                continue

            # Non-retryable error
            logger.error(f"TTS API error {response.status_code}: {response.text}")
            response.raise_for_status()

        # All retries exhausted
        logger.error(f"TTS failed after {MAX_RETRIES} retries for '{text}'")
        raise httpx.HTTPStatusError(
            f"TTS failed after {MAX_RETRIES} retries",
            request=response.request,
            response=response,
        )


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
