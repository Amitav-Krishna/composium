from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Genre(str, Enum):
    HIP_HOP = "hip-hop"
    POP = "pop"
    ROCK = "rock"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    LO_FI = "lo-fi"
    RNB = "rnb"


class Instrument(str, Enum):
    KICK = "kick"
    SNARE = "snare"
    HI_HAT = "hi-hat"
    PIANO = "piano"
    BASS = "bass"
    GUITAR = "guitar"
    SYNTH = "synth"
    CLAP = "clap"
    CYMBAL = "cymbal"
    STRINGS = "strings"


class MusicDescription(BaseModel):
    genre: Genre
    instruments: list[Instrument]
    tempo_bpm: Optional[int] = None
    mood: Optional[str] = None
    notes: Optional[str] = None


class QuantizedBeat(BaseModel):
    position: int  # 0-15 for 16th note grid within a bar
    bar: int = 0
    instrument: Instrument = Instrument.KICK
    velocity: float = 1.0  # 0.0-1.0


class RhythmPattern(BaseModel):
    beats: list[QuantizedBeat]
    bpm: int = 120
    bars: int = 1
    subdivisions: int = 16
    time_signature: str = "4/4"


class Layer(BaseModel):
    id: str  # UUID
    description: MusicDescription
    rhythm: RhythmPattern
    sample_mapping: dict[str, str]  # instrument_name -> file path
    audio_file: Optional[str] = None  # path to rendered layer audio


class Project(BaseModel):
    id: str  # UUID
    name: str
    layers: list[Layer] = []
    mixed_file: Optional[str] = None
    bpm: int = 120


# Request/Response models
class DescribeResponse(BaseModel):
    transcript: str
    description: MusicDescription


class RhythmResponse(BaseModel):
    rhythm: RhythmPattern
    detected_onsets: list[float]


class ProjectCreateRequest(BaseModel):
    name: str
    bpm: int = 120


class SpeakRequest(BaseModel):
    text: str
    voice_id: str = "jasmine"


class HealthResponse(BaseModel):
    status: str


class SampleCatalogResponse(BaseModel):
    genres: list[str]
    instruments: list[str]
    samples: dict[str, dict[str, list[str]]]  # genre -> instrument -> files
