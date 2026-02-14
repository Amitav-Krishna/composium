# Plan: Add Drums and Strum Guitar Instruments

## Context

Composium has piano and fingerpicking guitar. We're adding two more instruments following the established `arrange_<instrument>(analysis) -> Score` pattern.

**Key discovery:** abc2midi supports drums via regular notes on `%%MIDI channel 10` — each MIDI note number maps to a GM percussion sound. This fits our existing Voice/Note model with one small change to `notation.py`.

## Files to Modify/Create

1. **`composium/notation.py`** — add `midi_channel` field to `Voice`, update `score_to_abc` channel assignment
2. **`composium/instruments/drums.py`** (new) — drum arrangement logic
3. **`composium/instruments/strum_guitar.py`** (new) — strum guitar arrangement logic
4. **`composium/__init__.py`** — export both new functions

## Step 1: `notation.py` — Add `midi_channel` to Voice

Add an optional field to `Voice`:

```python
@dataclass
class Voice:
    notes: list[Note]
    name: str = "V1"
    midi_program: int = 0
    clef: str = "treble"
    midi_channel: int | None = None  # NEW: None = auto-assign
```

Update `score_to_abc` channel assignment (line 188):

```python
channel = voice.midi_channel if voice.midi_channel is not None else i
lines.append(f"%%MIDI channel {channel}")
```

This is backward-compatible — existing piano/guitar code doesn't set `midi_channel`, so it keeps auto-assigning.

## Step 2: `composium/instruments/drums.py` — Drum arrangement

`arrange_drums(analysis) -> Score` — one voice on MIDI channel 10.

**Pattern: Basic rock beat** (per measure in 4/4):
- Hi-hat (42): every eighth note (8 hits)
- Kick (36): beats 1 and 3
- Snare (38): beats 2 and 4

On beats where kick or snare overlaps hi-hat, they become simultaneous (ABC chord `[notes]`) — already handled by `_voice_to_abc_measures`.

**Helper:** `_rock_beat(target_measures, beats_per_measure) -> list[Note]`

Voice config: `clef="perc"`, `midi_channel=10`, `midi_program=0` (ignored on ch10).

## Step 3: `composium/instruments/strum_guitar.py` — Strum guitar

`arrange_strum_guitar(analysis) -> Score` — two voices, GM program 25.

**Voice 1: Lead melody** (same as fingerpicking guitar)
- Transpose to E2–E5 (MIDI 40–76), quantize, fill duration
- Copy `_quantize_to_grid` and `_transpose_to_range` (private helpers)

**Voice 2: Strummed chords**
- Derive chords from melody via `generate_chords()`
- For each chord, play all tones simultaneously (root + third + fifth) as quarter-note chords (4 per measure)
- Place in mid-guitar range (MIDI 48–64, roughly C3–E4)
- Distinct from piano Alberti (arpeggiated) and fingerpicking (ascending arpeggio)

**Helper:** `_strum(chords, beats_per_measure) -> list[Note]`

## Step 4: `__init__.py` — Exports

Add imports and `__all__` entries for `arrange_drums` and `arrange_strum_guitar`.

## Verification

```bash
python -c "
from composium import analyze, arrange_drums, arrange_strum_guitar, render
a = analyze('untitled.mp3')
d = arrange_drums(a)
render(d, 'drums_track.mp3')
s = arrange_strum_guitar(a)
render(s, 'strum_track.mp3')
"
ffprobe -v quiet -show_entries format=duration drums_track.mp3
ffprobe -v quiet -show_entries format=duration strum_track.mp3
```
