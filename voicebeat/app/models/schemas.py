from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


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
    VOCAL = "vocal"


class SegmentType(str, Enum):
    """Type of audio segment."""
    SPEECH = "speech"
    MELODY = "melody"   # Pitched musical content (humming, singing)
    RHYTHM = "rhythm"   # Unpitched percussive content (beatboxing, tapping)
    VOCAL = "vocal"     # Raw singing voice (preserved with autotune)


class AudioSegment(BaseModel):
    """A classified chunk of the user's recording."""
    id: str
    type: SegmentType
    start_seconds: float
    end_seconds: float
    transcript: Optional[str] = None  # Only for speech segments
    words: Optional[list[dict]] = None  # [{word, start, end}, ...] from STT
    audio_file: Optional[str] = None  # Path to extracted audio chunk


class PitchEvent(BaseModel):
    """A detected pitch in a melody segment."""
    time_seconds: float
    frequency_hz: float
    midi_note: int           # Converted MIDI note number
    note_name: str           # e.g., "C4", "A#3"
    duration_seconds: float
    confidence: float = 1.0


class MelodyContour(BaseModel):
    """Extracted melody from a hummed/sung segment."""
    pitches: list[PitchEvent] = []
    key_signature: Optional[str] = None  # e.g., "C major"
    abc_notation: Optional[str] = None   # ABC notation string
    tempo_bpm: Optional[int] = None


class QuantizedBeat(BaseModel):
    """A beat snapped to the nearest grid position."""
    position: int        # 0-15 for 16th note grid within a bar
    bar: int = 0
    instrument: Instrument = Instrument.KICK
    velocity: float = 1.0  # 0.0-1.0


class RhythmPattern(BaseModel):
    """Quantized rhythm grid extracted from beatboxing/tapping."""
    beats: list[QuantizedBeat] = []
    bpm: int = 120
    bars: int = 1
    subdivisions: int = 16
    time_signature: str = "4/4"


class MusicDescription(BaseModel):
    """Structured instructions extracted from speech."""
    genre: Optional[Genre] = None
    instruments: list[Instrument] = []
    tempo_bpm: Optional[int] = None
    mood: Optional[str] = None
    instructions: list[str] = []  # Raw instruction strings in order
    notes: Optional[str] = None


class Layer(BaseModel):
    """A single layer in a project."""
    id: str
    name: str = "Untitled Layer"
    instrument: Optional[Instrument] = None
    segment_type: SegmentType = SegmentType.RHYTHM
    rhythm: Optional[RhythmPattern] = None
    melody: Optional[MelodyContour] = None
    sample_mapping: dict[str, str] = {}
    audio_file: Optional[str] = None
    start_bar: int = 0  # Bar offset in the song (0 = from the start)
    created_at: datetime = Field(default_factory=datetime.now)


class Project(BaseModel):
    """A multi-layer music project."""
    id: str
    name: str = "Untitled Project"
    layers: list[Layer] = []
    segments: list[AudioSegment] = []  # Original segmented audio
    description: Optional[MusicDescription] = None
    mixed_file: Optional[str] = None
    bpm: int = 120
    key_signature: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


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


class ProcessResponse(BaseModel):
    """Response from the main /process endpoint."""
    project: Project
    feedback_text: str  # What the system understood (for TTS)
    feedback_audio: Optional[str] = None  # Path to TTS audio file


class EditLayerRequest(BaseModel):
    """Request to edit a layer via voice command."""
    audio_file: Optional[str] = None  # Path to voice command audio
    text_command: Optional[str] = None  # Or direct text command
