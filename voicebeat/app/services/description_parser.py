import json
import anthropic
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings
from app.models.schemas import MusicDescription, Genre, Instrument


PARSE_PROMPT = """You are a music description parser. Given a user's spoken description of the music they want, extract structured metadata.

Return ONLY valid JSON with the following fields:
- genre: one of "hip-hop", "pop", "rock", "jazz", "electronic", "lo-fi", "rnb"
- instruments: array of instruments from ["kick", "snare", "hi-hat", "piano", "bass", "guitar", "synth", "clap", "cymbal", "strings"]
- tempo_bpm: estimated tempo in BPM (integer, or null if not mentioned)
- mood: short description of the mood/vibe (string, or null)
- notes: any additional notes or requests (string, or null)

If the user doesn't specify a genre, make your best guess based on the instruments and mood.
If no instruments are mentioned, suggest appropriate ones for the genre.

Example input: "I want a chill lo-fi beat with piano and hi-hats"
Example output:
{
  "genre": "lo-fi",
  "instruments": ["piano", "hi-hat", "kick", "snare"],
  "tempo_bpm": 85,
  "mood": "chill, relaxed",
  "notes": null
}

User's description:
"""


async def parse_description(transcript: str) -> MusicDescription:
    """
    Parse a transcribed music description into structured metadata using Claude.

    Args:
        transcript: The transcribed text from the user's audio

    Returns:
        MusicDescription with genre, instruments, tempo, mood, notes
    """
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": PARSE_PROMPT + transcript,
            }
        ],
    )

    response_text = message.content[0].text.strip()

    # Try to extract JSON from the response
    try:
        # Handle case where response might have markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback to defaults
        return _get_default_description()

    return _parse_response(data)


def _parse_response(data: dict) -> MusicDescription:
    """Parse the JSON response into a MusicDescription model."""
    try:
        # Parse genre
        genre_str = data.get("genre", "pop").lower().replace("_", "-")
        try:
            genre = Genre(genre_str)
        except ValueError:
            genre = Genre.POP

        # Parse instruments
        instruments = []
        for inst in data.get("instruments", []):
            inst_str = inst.lower().replace("_", "-")
            try:
                instruments.append(Instrument(inst_str))
            except ValueError:
                pass

        # Ensure we have at least some instruments
        if not instruments:
            instruments = [Instrument.KICK, Instrument.SNARE, Instrument.HI_HAT]

        return MusicDescription(
            genre=genre,
            instruments=instruments,
            tempo_bpm=data.get("tempo_bpm"),
            mood=data.get("mood"),
            notes=data.get("notes"),
        )
    except Exception:
        return _get_default_description()


def _get_default_description() -> MusicDescription:
    """Return a sensible default description if parsing fails."""
    return MusicDescription(
        genre=Genre.POP,
        instruments=[Instrument.KICK, Instrument.SNARE, Instrument.HI_HAT, Instrument.BASS],
        tempo_bpm=120,
        mood="upbeat",
        notes=None,
    )
