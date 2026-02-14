# Vocal Processing Design: Speech-to-Musical-Vocals

## Problem

When users speak lyrics (rather than sing), the current vocal pipeline fails silently. The autotune path only works on pitched audio — speech has minimal pitch variation, so it produces empty or barely-audible output. Users want to say lyrics and have the system create musical vocals from them.

## Approach: LLM-Designed Vocal Contour

The vocal subagent upgrades from deterministic (no LLM) to LLM-backed. It receives the user's spoken words with timestamps, and the LLM designs either a melody (for pop/R&B) or beat placement (for hip-hop/electronic). Audio processing then applies per-word pitch-shifts or time-stretches to transform speech into musical vocals using the user's own voice.

## Two Modes

**Melodic mode** (pop, R&B, jazz, lo-fi): LLM assigns a MIDI note to each word/phrase. Audio processor pitch-shifts each word to its target note.

**Rhythmic mode** (hip-hop, electronic, rock): LLM assigns beat-grid positions to each word. Audio processor time-stretches and repositions each word to snap onto the grid.

The director picks the mode based on genre when delegating.

## Data Flow

```
User speaks lyrics
  -> Pulse STT (word-level timestamps)
  -> AudioSegment(type=SPEECH, transcript=..., words=[{word, start, end}, ...])
  -> Director sees speech segment with lyrics + word timestamps
  -> delegate_vocal(segment_id, mode="melodic"|"rhythmic")
  -> Vocal subagent (LLM) designs melody or rhythm placement per word
  -> Audio processor applies pitch-shifts / time-stretches per word
  -> Output MP3 layer
```

## Vocal Subagent Tools

Mode-specific tool sets (LLM only sees relevant tools):

**Melodic mode:**
- `design_vocal_melody` — assign MIDI notes + durations to words
- `render_vocal` — execute the design, produce audio
- `done` — signal completion

**Rhythmic mode:**
- `design_vocal_rhythm` — assign beat positions to words
- `render_vocal` — execute the design, produce audio
- `done` — signal completion

### LLM Context

The subagent receives:
- BPM, key, genre, mode
- Word list with timestamps and durations
- Scale notes (for melodic mode) or beat grid explanation (for rhythmic mode)
- Style notes from the director

## Audio Processing

### `pitch_shift_words()` (melodic mode)

For each word with an assigned MIDI note:
1. Extract word audio slice using STT timestamps
2. Detect current average pitch (librosa.pyin)
3. Calculate semitone shift (target - current), clamped to +/-7 semitones
4. Apply `librosa.effects.pitch_shift()`
5. Optionally time-stretch to match `duration_beats`
6. Crossfade adjacent words (32-64 samples)

### `beat_snap_words()` (rhythmic mode)

For each word with an assigned beat position:
1. Extract word audio slice using STT timestamps
2. Calculate target time: `(bar * 4 + beat_position/4) * seconds_per_beat`
3. Calculate available time slot until next word
4. Time-stretch word to fit slot (pydub/librosa)
5. Place at target position in silent buffer

### Post-processing (both modes)

- Normalize volume to match other layers
- Optional light reverb via numpy convolution

All processing uses existing libraries: librosa, pydub, numpy, soundfile.

## Files to Change

| File | Change |
|------|--------|
| `schemas.py` | Add `words: list[dict] \| None` field to `AudioSegment` |
| `segmenter.py` | Pass word timestamps through to `AudioSegment` |
| `vocal_processor.py` | Add `pitch_shift_words()` and `beat_snap_words()` functions |
| `subagents/vocal.py` | Rewrite as LLM-backed `BaseSubagent` with mode-specific tools |
| `director.py` | Update `delegate_vocal` to accept speech segments + mode param |

## Unchanged

- `routes.py`, `agent.py` shim, `composium_bridge.py`, `track_assembler.py`
- Existing autotune path still works for actual singing (melody segments)
- The director detects whether a segment is speech (lyrics to vocalize) vs melody (already singing) and routes accordingly

## Model Split

- Director: GPT-4o (OpenAI direct) — song planning + arrangement
- All subagents (drums, melody, vocal): Minimax m2.5 via OpenRouter — cheaper for mechanical work
