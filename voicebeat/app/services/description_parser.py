"""
Description Parser Service

Uses OpenAI to parse transcribed speech into structured music metadata.
"""

import json
from openai import AsyncOpenAI
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from app.models.schemas import MusicDescription, Genre, Instrument


# Initialize OpenAI client (created on first use)
_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


PARSE_PROMPT = """You are a music description parser. Given a user's spoken description of the music they want, extract structured metadata.

Return ONLY valid JSON with the following fields:
- genre: one of "hip-hop", "pop", "rock", "jazz", "electronic", "lo-fi", "rnb" (or null if not specified)
- instruments: array of instruments from ["kick", "snare", "hi-hat", "piano", "bass", "guitar", "synth", "clap", "cymbal", "strings", "vocal"]
- tempo_bpm: estimated tempo in BPM (integer, or null if not mentioned)
- mood: short description of the mood/vibe (string, or null)
- instructions: array of distinct instruction strings extracted from the speech
- notes: any additional notes or requests (string, or null)

If the user doesn't specify a genre, leave it as null.
Extract all distinct instructions/commands the user gives (e.g., "make this a piano", "add guitar", "make it faster").

Example input: "I want a chill lo-fi beat with piano and hi-hats, make it around 85 BPM"
Example output:
{
  "genre": "lo-fi",
  "instruments": ["piano", "hi-hat", "kick", "snare"],
  "tempo_bpm": 85,
  "mood": "chill, relaxed",
  "instructions": ["make a lo-fi beat", "use piano", "use hi-hats", "tempo around 85 BPM"],
  "notes": null
}

User's description:
"""


async def parse_description(transcript: str) -> MusicDescription:
    """
    Parse a transcribed music description into structured metadata using OpenAI.

    Args:
        transcript: The transcribed text from the user's audio

    Returns:
        MusicDescription with genre, instruments, tempo, mood, instructions, notes
    """
    client = _get_client()

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": PARSE_PROMPT + transcript,
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    response_text = response.choices[0].message.content.strip()

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return _get_default_description()

    return _parse_response(data)


def _parse_response(data: dict) -> MusicDescription:
    """Parse the JSON response into a MusicDescription model."""
    try:
        # Parse genre
        genre = None
        genre_str = data.get("genre")
        if genre_str:
            genre_str = genre_str.lower().replace("_", "-")
            try:
                genre = Genre(genre_str)
            except ValueError:
                genre = None

        # Parse instruments
        instruments = []
        for inst in data.get("instruments", []):
            inst_str = inst.lower().replace("_", "-")
            try:
                instruments.append(Instrument(inst_str))
            except ValueError:
                pass

        # Parse instructions
        instructions = data.get("instructions", [])
        if not isinstance(instructions, list):
            instructions = [instructions] if instructions else []

        return MusicDescription(
            genre=genre,
            instruments=instruments,
            tempo_bpm=data.get("tempo_bpm"),
            mood=data.get("mood"),
            instructions=instructions,
            notes=data.get("notes"),
        )
    except Exception:
        return _get_default_description()


def _get_default_description() -> MusicDescription:
    """Return a sensible default description if parsing fails."""
    return MusicDescription(
        genre=None,
        instruments=[],
        tempo_bpm=None,
        mood=None,
        instructions=[],
        notes=None,
    )


async def extract_instructions(transcripts: list[str]) -> MusicDescription:
    """
    Extract combined instructions from multiple speech transcripts.

    Args:
        transcripts: List of speech transcript strings

    Returns:
        Combined MusicDescription from all instructions
    """
    combined = "\n".join(transcripts)
    return await parse_description(combined)
