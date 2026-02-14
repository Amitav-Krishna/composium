import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import Genre, Instrument, MusicDescription
from app.services.description_parser import _parse_response, _get_default_description


class TestParseResponse:
    """Tests for the _parse_response helper function."""

    def test_parse_valid_response(self):
        """Test parsing a valid JSON response."""
        data = {
            "genre": "hip-hop",
            "instruments": ["kick", "snare", "hi-hat", "bass"],
            "tempo_bpm": 90,
            "mood": "chill, laid back",
            "notes": None,
        }

        result = _parse_response(data)

        assert result.genre == Genre.HIP_HOP
        assert Instrument.KICK in result.instruments
        assert Instrument.SNARE in result.instruments
        assert Instrument.HI_HAT in result.instruments
        assert Instrument.BASS in result.instruments
        assert result.tempo_bpm == 90
        assert result.mood == "chill, laid back"

    def test_parse_lo_fi_genre(self):
        """Test parsing lo-fi genre with hyphen."""
        data = {
            "genre": "lo-fi",
            "instruments": ["piano", "kick"],
            "tempo_bpm": 85,
            "mood": "relaxed",
            "notes": None,
        }

        result = _parse_response(data)
        assert result.genre == Genre.LO_FI

    def test_parse_unknown_genre_defaults_to_pop(self):
        """Test that unknown genre defaults to pop."""
        data = {
            "genre": "unknown-genre",
            "instruments": ["kick"],
            "tempo_bpm": 120,
            "mood": None,
            "notes": None,
        }

        result = _parse_response(data)
        assert result.genre == Genre.POP

    def test_parse_empty_instruments_adds_defaults(self):
        """Test that empty instruments list gets defaults."""
        data = {
            "genre": "pop",
            "instruments": [],
            "tempo_bpm": 120,
            "mood": None,
            "notes": None,
        }

        result = _parse_response(data)
        assert len(result.instruments) > 0
        assert Instrument.KICK in result.instruments

    def test_parse_invalid_instruments_are_filtered(self):
        """Test that invalid instruments are filtered out."""
        data = {
            "genre": "rock",
            "instruments": ["kick", "invalid_instrument", "snare"],
            "tempo_bpm": 140,
            "mood": None,
            "notes": None,
        }

        result = _parse_response(data)
        assert Instrument.KICK in result.instruments
        assert Instrument.SNARE in result.instruments
        assert len(result.instruments) == 2


class TestGetDefaultDescription:
    """Tests for the _get_default_description function."""

    def test_default_description_has_required_fields(self):
        """Test that default description has all required fields."""
        result = _get_default_description()

        assert isinstance(result, MusicDescription)
        assert result.genre is not None
        assert len(result.instruments) > 0
        assert result.tempo_bpm is not None

    def test_default_description_has_basic_instruments(self):
        """Test that default description has basic drum instruments."""
        result = _get_default_description()

        assert Instrument.KICK in result.instruments
        assert Instrument.SNARE in result.instruments
