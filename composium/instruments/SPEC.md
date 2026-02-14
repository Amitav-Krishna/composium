# Composium — Music Toolkit for Agents

## Overview

Composium is the backend for an app where agents create music from user audio input (humming, lyrics, etc.). The toolkit gives agents programmatic, non-hallucinating ways to analyze audio, generate arrangements, and render tracks.

First iteration: Python library, piano instrument only, designed for easy extension to drums/bass/guitar later.

## Dependencies

### Python (pip)
- `librosa` — audio analysis (tempo, pitch, key, beats)
- `numpy`, `scipy` — numerical (installed with librosa)

### System
- `abc2midi` — ABC notation to MIDI conversion
- `timidity` — MIDI to WAV synthesis
- `ffmpeg` — WAV to MP3 encoding

## Package Structure

```
composium/
├── __init__.py          # Public API exports
├── analysis.py          # Audio analysis (tempo, pitch, key, beats)
├── notation.py          # Musical data types + ABC notation generation
├── arrangement.py       # Chord & accompaniment generation
├── instruments/
│   ├── __init__.py
│   └── piano.py         # Piano-specific arrangement patterns
├── render.py            # ABC → MIDI → MP3 pipeline
```

## Public API

```python
from composium import analyze, arrange_piano, render

# Step 1: Analyze user's audio
analysis = analyze("humming.mp3")
# Returns: Analysis(tempo, key, duration, notes, beats)

# Step 2: Generate piano arrangement
score = arrange_piano(analysis)
# Returns: Score(voices=[Voice(melody), Voice(chords)], tempo, key, duration)

# Step 3: Render to audio
render(score, "soundtrack.mp3")
# Produces: soundtrack.abc, soundtrack.mid, soundtrack.mp3
```

## Module Specs

### `notation.py` — Musical Data Types & ABC Generation

#### Data Types

```python
@dataclass
class Note:
    midi_pitch: int        # MIDI note number (60 = middle C)
    start_beat: float      # Beat position (0-indexed)
    duration_beats: float  # Length in beats

@dataclass
class Voice:
    notes: list[Note]
    name: str              # e.g. "RH", "LH"
    midi_program: int      # GM instrument (0 = piano)
    clef: str              # "treble" or "bass"

@dataclass
class Score:
    voices: list[Voice]
    tempo: int             # BPM
    key: str               # e.g. "Em", "C", "F#m"
    time_sig: tuple[int, int]  # e.g. (4, 4)
    duration: float        # Total duration in seconds

@dataclass
class Analysis:
    tempo: int
    key: str
    duration: float        # Input audio duration in seconds
    notes: list[Note]      # Detected melody notes
    beats: list[float]     # Beat timestamps in seconds
```

#### Functions

- **`midi_to_abc(midi_pitch: int, key_sig: str) -> str`**
  Converts a MIDI note number to an ABC pitch string, respecting the key signature (e.g., MIDI 64 in key of Em → `e` not `^d`).

- **`score_to_abc(score: Score) -> str`**
  Renders a full `Score` to a complete ABC notation string with headers (`X:`, `T:`, `M:`, `L:`, `Q:`, `K:`), voice definitions, and `%%MIDI program` directives.

#### Key Signature Handling

Map each key to its sharps/flats set so `midi_to_abc` knows which accidentals are implicit vs. explicit. Support all major and minor keys.

### `analysis.py` — Audio Analysis

#### Function

- **`analyze(audio_path: str) -> Analysis`**

#### Algorithm

1. Load audio with `librosa.load()` (mono, sr=22050)
2. Detect tempo and beats via `librosa.beat.beat_track()`
3. Extract pitch contour via `librosa.pyin(fmin=80, fmax=600)` — best for humming/singing
4. Quantize detected pitches to nearest semitone (round MIDI floats to int)
5. Segment continuous pitch regions into `Note` events, merging consecutive frames with the same pitch
6. Detect key via pitch-class histogram: count occurrences of each pitch class (0-11), correlate against major and minor scale templates for all 12 roots, pick the best match
7. Return `Analysis(tempo, key, duration, notes, beats)`

### `arrangement.py` — Arrangement Engine

#### Functions

- **`generate_chords(notes: list[Note], key: str, beats_per_measure: int) -> list[Chord]`**

  Infers a chord progression from melody notes using scale-degree heuristics:
  - Group melody notes by measure
  - For each measure, count pitch classes present
  - Score candidate chords (I, ii, iii, IV, V, vi, vii°) by how many melody pitches they contain
  - Pick the highest-scoring chord per measure
  - Return list of `Chord(root_midi, quality, start_beat, duration_beats)`

- **`fill_duration(melody_notes: list[Note], target_measures: int, tempo: int) -> list[Note]`**

  Pads or repeats the melody to fill the target number of measures:
  - Calculate how many measures the melody currently spans
  - If shorter than target: loop the melody, offsetting `start_beat` each repetition
  - If longer: truncate
  - Pad remaining space with rests (gaps in note list)

### `instruments/piano.py` — Piano Patterns

#### Function

- **`arrange_piano(analysis: Analysis) -> Score`**

#### Algorithm

1. Calculate total measures from `analysis.duration` and `analysis.tempo`
2. Transpose melody notes up 1 octave (+12 MIDI) into comfortable piano range (C4-C6)
3. Quantize note start/duration to eighth-note grid (0.5-beat resolution)
4. Call `fill_duration()` to match target measure count
5. Call `generate_chords()` to get chord progression
6. Generate left-hand Alberti bass pattern from chords:
   - For each chord, arpeggiate root-fifth-third-fifth in eighth notes
   - Place in bass range (C2-C4)
7. Assemble 2-voice `Score`:
   - Voice 1 (RH): melody, treble clef, MIDI program 0
   - Voice 2 (LH): Alberti bass, bass clef, MIDI program 0

### `render.py` — Rendering Pipeline

#### Function

- **`render(score: Score, output_path: str, keep_intermediates: bool = True)`**

#### Pipeline

1. Call `score_to_abc(score)` → write to `<stem>.abc`
2. Run `abc2midi <stem>.abc -o <stem>.mid` via subprocess
3. Run `timidity <stem>.mid -Ow -o - | ffmpeg -i - -y <stem>.mp3` via subprocess
4. Verify output MP3 duration is within ±2s of `score.duration` using ffprobe
5. If `keep_intermediates` is False, delete `.abc` and `.mid` files

## Design Decisions

1. **ABC as intermediate format** — Human-readable, debuggable, handled by proven `abc2midi` tool. Agents never write ABC directly; the library generates it programmatically.

2. **Duration matching** — Output should match input audio duration (±2s tolerance). `fill_duration()` calculates total measures from input duration and tempo, then loops or pads the melody.

3. **Extensibility** — Adding a new instrument means adding a file in `instruments/` with an `arrange_<instrument>(analysis) -> Score` function. The `Score` supports multiple voices with different MIDI programs.

4. **No hallucination** — All musical content is derived from actual audio analysis data. No generated content is invented from scratch.

5. **Quantization** — Eighth-note grid (0.5-beat resolution) balances rhythmic fidelity with clean notation.

## Verification

```bash
source .venv/bin/activate
python -c "
from composium import analyze, arrange_piano, render
a = analyze('untitled.mp3')
print(f'Tempo: {a.tempo}, Key: {a.key}, Duration: {a.duration:.1f}s, Notes: {len(a.notes)}')
s = arrange_piano(a)
render(s, 'soundtrack.mp3')
"
# Verify: soundtrack.mp3 duration ≈ untitled.mp3 duration (6.87s ± 2s)
ffprobe -v quiet -show_entries format=duration soundtrack.mp3
```
