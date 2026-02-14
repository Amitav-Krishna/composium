import httpx
from pathlib import Path
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings


WAVES_TTS_URL = "https://waves-api.smallest.ai/api/v1/lightning-v2/get_speech"

_word_cache: dict[tuple[str, str], bytes] = {}


async def speak_word(word: str, voice_id: str | None = None) -> bytes:
    """Cached single-word TTS. Avoids redundant API calls for repeated words."""
    cache_key = (word.lower().strip(), voice_id or settings.tts_voice_id)
    if cache_key in _word_cache:
        return _word_cache[cache_key]
    audio = await speak(word, voice_id)
    _word_cache[cache_key] = audio
    return audio


async def speak(
    text: str,
    voice_id: str | None = None,
    sample_rate: int | None = None,
) -> bytes:
    """
    Convert text to speech using Smallest.ai Waves Lightning TTS.

    Args:
        text: The text to convert to speech
        voice_id: Voice ID to use (default: from settings)
        sample_rate: Audio sample rate (default: from settings)

    Returns:
        Raw WAV audio bytes
    """
    voice = voice_id or settings.tts_voice_id
    rate = sample_rate or settings.tts_sample_rate

    headers = {
        "Authorization": f"Bearer {settings.smallest_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "voice_id": voice,
        "text": text,
        "sample_rate": rate,
        "add_wav_header": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            WAVES_TTS_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.content


async def speak_to_file(
    text: str,
    output_path: str | Path,
    voice_id: str | None = None,
    sample_rate: int | None = None,
) -> Path:
    """
    Convert text to speech and save to a file.

    Args:
        text: The text to convert to speech
        output_path: Path to save the audio file
        voice_id: Voice ID to use
        sample_rate: Audio sample rate

    Returns:
        Path to the saved audio file
    """
    audio_bytes = await speak(text, voice_id, sample_rate)
    output = Path(output_path)
    output.write_bytes(audio_bytes)
    return output
