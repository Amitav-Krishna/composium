"""Musical data types and ABC notation generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Note:
    midi_pitch: int  # MIDI note number (60 = middle C)
    start_beat: float  # Beat position (0-indexed)
    duration_beats: float  # Length in beats


@dataclass
class Chord:
    root_midi: int  # Root note MIDI number
    quality: str  # "major", "minor", "dim"
    start_beat: float
    duration_beats: float


@dataclass
class Voice:
    notes: list[Note]
    name: str = "V1"
    midi_program: int = 0  # GM instrument (0 = piano)
    clef: str = "treble"
    midi_channel: int | None = None  # Override auto-assigned channel (e.g. 10 for drums)


@dataclass
class Score:
    voices: list[Voice]
    tempo: int  # BPM
    key: str  # e.g. "Em", "C", "F#m"
    time_sig: tuple[int, int] = (4, 4)
    duration: float = 0.0  # Total duration in seconds


@dataclass
class Analysis:
    tempo: int
    key: str
    duration: float  # Input audio duration in seconds
    notes: list[Note] = field(default_factory=list)
    beats: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Key-signature tables
# ---------------------------------------------------------------------------

# Sharps/flats present in each key signature as pitch classes (0-11, C=0).
# Used to decide whether an accidental is "in key" or needs explicit markup.

_SHARP_ORDER = [6, 1, 8, 3, 10, 5, 0]  # F# C# G# D# A# E# B#
_FLAT_ORDER = [10, 3, 8, 1, 6, 11, 4]  # Bb Eb Ab Db Gb Cb Fb

# Map key name -> number of sharps (+) or flats (-).
_KEY_SIG_MAP: dict[str, int] = {
    # Major keys
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "Gb": -6,
    "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1,
    # Minor keys (relative minor shares signature with its relative major)
    "Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5,
    "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6,
}


def _sharps_in_key(key: str) -> set[int]:
    """Return the set of pitch-classes that are sharped in *key*."""
    n = _KEY_SIG_MAP.get(key, 0)
    if n > 0:
        return set(_SHARP_ORDER[:n])
    return set()


def _flats_in_key(key: str) -> set[int]:
    """Return the set of pitch-classes that are flatted in *key*."""
    n = _KEY_SIG_MAP.get(key, 0)
    if n < 0:
        return set(_FLAT_ORDER[: -n])
    return set()


# ---------------------------------------------------------------------------
# MIDI ↔ ABC conversion
# ---------------------------------------------------------------------------

# ABC note names for each pitch class when using sharps vs flats.
_SHARP_NAMES = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
_FLAT_NAMES = ["C", "_D", "D", "_E", "E", "F", "_G", "G", "_A", "A", "_B", "B"]
_NATURAL_NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]


def midi_to_abc(midi_pitch: int, key_sig: str = "C") -> str:
    """Convert a MIDI note number to an ABC pitch string.

    Respects the key signature so that notes already sharped/flatted by the
    key don't get redundant accidentals, while chromatic alterations do.
    """
    pc = midi_pitch % 12  # pitch class 0-11
    octave = midi_pitch // 12 - 1  # MIDI octave (middle-C = octave 4)

    sharps = _sharps_in_key(key_sig)
    flats = _flats_in_key(key_sig)

    if sharps:
        # Key has sharps
        if pc in sharps:
            # This pitch class is sharped by key sig — write natural name
            name = _NATURAL_NAMES[pc]
        else:
            name = _SHARP_NAMES[pc]
    elif flats:
        if pc in flats:
            name = _NATURAL_NAMES[pc]
        else:
            name = _FLAT_NAMES[pc]
    else:
        name = _SHARP_NAMES[pc]

    # ABC octave encoding: C-B in octave 4 = "C" to "B"
    # Octave 5 = "c" to "b", octave 6 = "c'" etc.
    # Octave 3 = "C," to "B,", octave 2 = "C,," etc.
    base_letter = name[-1] if name[-1].isalpha() else name[name.index("^") + 1:] if "^" in name else name[name.index("_") + 1:]
    prefix = name[: -len(base_letter)]

    if octave >= 5:
        base_letter = base_letter.lower()
        suffix = "'" * (octave - 5)
    elif octave == 4:
        base_letter = base_letter.upper()
        suffix = ""
    else:
        base_letter = base_letter.upper()
        suffix = "," * (4 - octave)

    return prefix + base_letter + suffix


def _duration_to_abc(duration_beats: float, base_length: float = 0.5) -> str:
    """Convert a beat duration to ABC length modifier.

    With L:1/8 (base_length=0.5 beats), a quarter note (1 beat) = "2",
    eighth note (0.5) = "" (empty), half note (2) = "4", etc.
    """
    ratio = duration_beats / base_length
    if ratio <= 0:
        return ""
    # Express as integer or simple fraction
    ratio = round(ratio * 2) / 2  # snap to half-units
    if ratio == 1.0:
        return ""
    if ratio == int(ratio):
        return str(int(ratio))
    # Handle dotted / fractional durations
    num = int(ratio * 2)
    return f"{num}/2"


# ---------------------------------------------------------------------------
# Score → ABC
# ---------------------------------------------------------------------------

def score_to_abc(score: Score) -> str:
    """Render a Score to a complete ABC notation string."""
    num, den = score.time_sig
    lines: list[str] = [
        "X:1",
        "T:Composium Soundtrack",
        f"M:{num}/{den}",
        "L:1/8",
        f"Q:1/4={score.tempo}",
        f"K:{score.key}",
    ]

    beats_per_measure = num * (4 / den)  # e.g. 4/4 → 4 beats

    for i, voice in enumerate(score.voices, 1):
        lines.append(f"V:{i} clef={voice.clef} name=\"{voice.name}\"")
        lines.append(f"%%MIDI program {voice.midi_program}")
        channel = voice.midi_channel if voice.midi_channel is not None else i
        lines.append(f"%%MIDI channel {channel}")

        abc_measures = _voice_to_abc_measures(voice, score.key, beats_per_measure)
        # Join measures with bar lines, line break every 4 measures
        for j in range(0, len(abc_measures), 4):
            chunk = abc_measures[j : j + 4]
            suffix = " |" if j + 4 < len(abc_measures) else " ||"
            lines.append(" | ".join(chunk) + suffix)

    return "\n".join(lines) + "\n"


def _voice_to_abc_measures(voice: Voice, key: str, beats_per_measure: float) -> list[str]:
    """Convert a Voice's notes into a list of ABC measure strings."""
    if not voice.notes:
        return ["z8"]

    # Figure out total measures needed
    max_beat = max(n.start_beat + n.duration_beats for n in voice.notes)
    total_measures = max(1, math.ceil(max_beat / beats_per_measure))

    base = 0.5  # L:1/8 = half a beat
    measures: list[str] = []

    for m_idx in range(total_measures):
        m_start = m_idx * beats_per_measure
        m_end = m_start + beats_per_measure

        # Collect notes that fall in this measure, clipped to boundaries
        m_notes: list[Note] = []
        for n in voice.notes:
            n_end = n.start_beat + n.duration_beats
            if n_end > m_start and n.start_beat < m_end:
                clipped_start = max(n.start_beat, m_start)
                clipped_end = min(n_end, m_end)
                m_notes.append(Note(n.midi_pitch, clipped_start, clipped_end - clipped_start))

        m_notes.sort(key=lambda n: n.start_beat)

        # Group concurrent notes (same start beat) into chords
        groups: list[list[Note]] = []
        for n in m_notes:
            if groups and abs(n.start_beat - groups[-1][0].start_beat) < 0.01:
                groups[-1].append(n)
            else:
                groups.append([n])

        tokens: list[str] = []
        cursor = m_start

        for group in groups:
            start = group[0].start_beat
            # Use the shortest duration in the group for timing
            dur = min(n.duration_beats for n in group)

            # Rest before this group?
            gap = start - cursor
            if gap > 0.01:
                tokens.append("z" + _duration_to_abc(gap, base))

            abc_dur = _duration_to_abc(dur, base)
            if len(group) == 1:
                abc_pitch = midi_to_abc(group[0].midi_pitch, key)
                tokens.append(abc_pitch + abc_dur)
            else:
                # ABC chord: [CEG]duration
                pitches = [midi_to_abc(n.midi_pitch, key) for n in group]
                tokens.append("[" + "".join(pitches) + "]" + abc_dur)

            cursor = start + dur

        # Trailing rest to fill the measure
        gap = m_end - cursor
        if gap > 0.01:
            tokens.append("z" + _duration_to_abc(gap, base))

        measures.append("".join(tokens))

    return measures
