import io
import sys
from pathlib import Path

import httpx
from pydub import AudioSegment

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings

WAVES_TTS_URL = "https://waves-api.smallest.ai/api/v1/lightning/get_speech"


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


async def speak_mp3(
    text: str,
    voice_id: str | None = None,
    sample_rate: int | None = None,
) -> bytes:
    """
    Convert text to speech using Smallest.ai Waves Lightning TTS,
    then transcode WAV bytes to MP3.

    Returns:
        Raw MP3 audio bytes
    """
    wav_bytes = await speak(text, voice_id, sample_rate)
    audio_segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3")
    return mp3_buffer.getvalue()


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
