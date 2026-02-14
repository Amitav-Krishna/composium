"""
Notation Service

ABC notation builder and optional MIDI export.
ABC is a human-readable text format for music notation.
"""

from typing import Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schemas import MelodyContour, PitchEvent, RhythmPattern
from app.utils.music_theory import NOTE_NAMES, midi_to_note_name

# ABC note names for different octaves
# Octave 4: C D E F G A B
# Octave 5: c d e f g a b
# Octave 3: C, D, E, etc.
# Octave 6: c' d' e' etc.


def melody_to_abc(
    melody: MelodyContour,
    title: str = "User Melody",
    tempo: Optional[int] = None,
    time_sig: str = "4/4",
) -> str:
    """
    Convert a MelodyContour to ABC notation.

    Args:
        melody: The melody contour with pitch events
        title: Title for the piece
        tempo: Tempo in BPM (uses melody's tempo if not provided)
        time_sig: Time signature (default 4/4)

    Returns:
        ABC notation string
    """
    bpm = tempo or melody.tempo_bpm or 120
    key = _parse_key(melody.key_signature) if melody.key_signature else "C"

    lines = [
        "X:1",
        f"T:{title}",
        f"M:{time_sig}",
        "L:1/8",  # Default note length is eighth note
        f"Q:1/4={bpm}",
        f"K:{key}",
    ]

    # Convert pitch events to ABC notes
    if melody.pitches:
        abc_notes = [_pitch_event_to_abc(p, bpm) for p in melody.pitches]
        melody_line = " ".join(abc_notes)
        lines.append(melody_line + " |]")
    else:
        lines.append("z4 |]")  # Empty measure with rests

    return "\n".join(lines)


def rhythm_to_abc(
    rhythm: RhythmPattern,
    title: str = "User Rhythm",
    instrument: str = "percussion",
) -> str:
    """
    Convert a RhythmPattern to ABC notation (percussion).

    For percussion, we use a single pitch (typically B in ABC)
    and note the rhythm only.

    Args:
        rhythm: The rhythm pattern with quantized beats
        title: Title for the piece
        instrument: Instrument name

    Returns:
        ABC notation string
    """
    lines = [
        "X:1",
        f"T:{title}",
        f"M:{rhythm.time_signature}",
        "L:1/16",  # Use 16th notes as base
        f"Q:1/4={rhythm.bpm}",
        "K:C clef=perc",  # Percussion clef
    ]

    # Build rhythm pattern
    # Create a grid of 16th notes, marking where beats occur
    for bar in range(rhythm.bars):
        bar_beats = [b for b in rhythm.beats if b.bar == bar]
        positions_with_beats = {b.position for b in bar_beats}

        notes = []
        i = 0
        while i < rhythm.subdivisions:
            if i in positions_with_beats:
                notes.append("B")  # Use B for percussion hits
            else:
                notes.append("z")  # Rest
            i += 1

        bar_str = " ".join(notes) + " |"
        lines.append(bar_str)

    # Close the piece
    lines[-1] = lines[-1][:-1] + "]"  # Replace final | with ]

    return "\n".join(lines)


def _pitch_event_to_abc(pitch: PitchEvent, bpm: int) -> str:
    """Convert a single pitch event to ABC notation."""
    midi = pitch.midi_note
    octave = (midi // 12) - 1
    note_index = midi % 12
    note_name = NOTE_NAMES[note_index]

    # Handle sharps
    if "#" in note_name:
        abc_note = "^" + note_name[0]
    else:
        abc_note = note_name

    # Handle octaves
    # ABC uses:
    # C, D, E, F, G, A, B = octave 4
    # c, d, e, f, g, a, b = octave 5
    # C,, D,, = octave 2
    # c', d' = octave 6

    if octave < 4:
        abc_note = abc_note + "," * (4 - octave)
    elif octave == 5:
        abc_note = abc_note.lower()
    elif octave > 5:
        abc_note = abc_note.lower() + "'" * (octave - 5)

    # Calculate duration
    # Base unit is 1/8 note, so:
    # 1/8 note = no modifier
    # 1/4 note = "2"
    # 1/2 note = "4"
    # 1/16 note = "/2"

    duration_beats = pitch.duration_seconds * (bpm / 60.0)

    # Convert to eighth-note units
    eighth_units = duration_beats * 2

    if eighth_units >= 4:
        abc_note += "4"
    elif eighth_units >= 2:
        abc_note += "2"
    elif eighth_units >= 1:
        pass  # Default eighth note
    elif eighth_units >= 0.5:
        abc_note += "/2"
    else:
        abc_note += "/4"

    return abc_note


def _parse_key(key_sig: Optional[str]) -> str:
    """Parse a key signature string to ABC format."""
    if not key_sig:
        return "C"

    parts = key_sig.replace("-", " ").split()
    if len(parts) >= 2:
        root = parts[0].replace("#", "^").replace("b", "_")
        mode = parts[1].lower()
        if mode == "minor":
            return root + "m"
        return root
    elif len(parts) == 1:
        return parts[0].replace("#", "^").replace("b", "_")

    return "C"


def create_score_abc(
    title: str,
    tempo: int,
    key: str,
    time_sig: str,
    voices: list[tuple[str, str]],  # List of (voice_name, abc_content)
) -> str:
    """
    Create a multi-voice ABC score.

    Args:
        title: Title of the piece
        tempo: Tempo in BPM
        key: Key signature
        time_sig: Time signature
        voices: List of (voice_name, abc_content) tuples

    Returns:
        Complete ABC notation string
    """
    lines = [
        "X:1",
        f"T:{title}",
        f"M:{time_sig}",
        "L:1/8",
        f"Q:1/4={tempo}",
        f"K:{key}",
    ]

    # Voice definitions
    for i, (name, _) in enumerate(voices):
        lines.append(f"V:{i+1} name=\"{name}\"")

    lines.append("")

    # Voice content
    for i, (name, content) in enumerate(voices):
        lines.append(f"V:{i+1}")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


# Optional: MIDI export (requires midiutil)
def melody_to_midi(
    melody: MelodyContour,
    output_path: str,
    tempo: Optional[int] = None,
    instrument: int = 0,  # 0 = Piano
) -> str:
    """
    Convert a MelodyContour to MIDI file.

    Args:
        melody: The melody contour with pitch events
        output_path: Path to save the MIDI file
        tempo: Tempo in BPM
        instrument: MIDI instrument number (0-127)

    Returns:
        Path to the saved MIDI file
    """
    try:
        from midiutil import MIDIFile
    except ImportError:
        raise ImportError("midiutil is required for MIDI export. Install with: pip install midiutil")

    bpm = tempo or melody.tempo_bpm or 120

    # Create MIDI file with 1 track
    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    midi.addProgramChange(0, 0, 0, instrument)

    # Convert seconds to beats for MIDI
    for pitch in melody.pitches:
        start_beat = pitch.time_seconds * (bpm / 60.0)
        duration_beats = pitch.duration_seconds * (bpm / 60.0)
        velocity = int(pitch.confidence * 100)  # Use confidence as velocity

        midi.addNote(
            track=0,
            channel=0,
            pitch=pitch.midi_note,
            time=start_beat,
            duration=duration_beats,
            volume=velocity,
        )

    # Write to file
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return output_path
