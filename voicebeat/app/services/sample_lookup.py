import random
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings
from app.models.schemas import MusicDescription, Genre, Instrument


# Genre fallback chains - if samples not found in primary genre, try these
GENRE_FALLBACKS = {
    Genre.HIP_HOP: [Genre.RNB, Genre.ELECTRONIC, Genre.POP],
    Genre.POP: [Genre.ELECTRONIC, Genre.ROCK],
    Genre.ROCK: [Genre.POP, Genre.JAZZ],
    Genre.JAZZ: [Genre.LO_FI, Genre.RNB],
    Genre.ELECTRONIC: [Genre.POP, Genre.HIP_HOP],
    Genre.LO_FI: [Genre.JAZZ, Genre.HIP_HOP],
    Genre.RNB: [Genre.HIP_HOP, Genre.JAZZ, Genre.POP],
}


def lookup_samples(
    description: MusicDescription,
    samples_dir: Optional[Path] = None,
) -> dict[str, str]:
    """
    Look up sample files for each instrument in the description.

    Args:
        description: MusicDescription with genre and instruments
        samples_dir: Path to samples directory (default: from settings)

    Returns:
        Dict mapping instrument names to file paths
    """
    base_dir = samples_dir or settings.samples_dir
    base_dir = Path(base_dir)

    result = {}

    for instrument in description.instruments:
        sample_path = _find_sample(base_dir, description.genre, instrument)
        if sample_path:
            result[instrument.value] = str(sample_path)

    return result


def _find_sample(
    base_dir: Path,
    genre: Genre,
    instrument: Instrument,
) -> Optional[Path]:
    """Find a sample file for the given genre and instrument."""
    # Try the primary genre first
    genres_to_try = [genre] + GENRE_FALLBACKS.get(genre, [])

    for try_genre in genres_to_try:
        genre_dir = base_dir / try_genre.value / instrument.value

        if genre_dir.exists():
            # Find all .wav files
            wav_files = list(genre_dir.glob("*.wav"))
            if wav_files:
                # Pick a random sample for variety
                return random.choice(wav_files)

    # If still not found, try any genre
    for genre_dir in base_dir.iterdir():
        if genre_dir.is_dir():
            inst_dir = genre_dir / instrument.value
            if inst_dir.exists():
                wav_files = list(inst_dir.glob("*.wav"))
                if wav_files:
                    return random.choice(wav_files)

    return None


def get_sample_catalog(samples_dir: Optional[Path] = None) -> dict:
    """
    Get a catalog of all available samples.

    Returns:
        Dict with genres, instruments, and nested sample listings
    """
    base_dir = Path(samples_dir or settings.samples_dir)

    genres = []
    instruments = set()
    samples = {}

    if not base_dir.exists():
        return {"genres": [], "instruments": [], "samples": {}}

    for genre_dir in sorted(base_dir.iterdir()):
        if genre_dir.is_dir() and not genre_dir.name.startswith("."):
            genre_name = genre_dir.name
            genres.append(genre_name)
            samples[genre_name] = {}

            for inst_dir in sorted(genre_dir.iterdir()):
                if inst_dir.is_dir() and not inst_dir.name.startswith("."):
                    inst_name = inst_dir.name
                    instruments.add(inst_name)

                    wav_files = [f.name for f in inst_dir.glob("*.wav")]
                    if wav_files:
                        samples[genre_name][inst_name] = sorted(wav_files)

    return {
        "genres": genres,
        "instruments": sorted(instruments),
        "samples": samples,
    }
