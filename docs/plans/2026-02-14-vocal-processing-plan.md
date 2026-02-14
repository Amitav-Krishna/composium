# Speech-to-Musical-Vocals Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform spoken lyrics into musical vocals — melodic singing (pop/R&B) or beat-snapped speech (hip-hop) — using the user's own voice.

**Architecture:** Speech segments carry word-level timestamps from STT through to a new LLM-backed vocal subagent. The subagent designs a melody or beat placement per word, then audio processing applies per-word pitch-shifts or time-stretches. Two modes: melodic (pitch-shift to MIDI notes) and rhythmic (snap words to beat grid).

**Tech Stack:** librosa (pitch_shift, pyin, time_stretch), pydub (export), numpy (buffer manipulation), OpenRouter/Minimax m2.5 (vocal subagent LLM), OpenAI/GPT-4o (director only).

**Design doc:** `docs/plans/2026-02-14-vocal-processing-design.md`

---

### Task 1: Add `words` field to AudioSegment schema

**Files:**
- Modify: `voicebeat/app/models/schemas.py:39-46`

**Step 1: Add the field**

In `AudioSegment`, add a `words` field after `transcript`:

```python
class AudioSegment(BaseModel):
    """A classified chunk of the user's recording."""
    id: str
    type: SegmentType
    start_seconds: float
    end_seconds: float
    transcript: Optional[str] = None  # Only for speech segments
    words: Optional[list[dict]] = None  # [{word, start, end}, ...] from STT
    audio_file: Optional[str] = None  # Path to extracted audio chunk
```

**Step 2: Commit**

```bash
git add voicebeat/app/models/schemas.py
git commit -m "feat: add words field to AudioSegment for word-level timestamps"
```

---

### Task 2: Thread word timestamps through the segmenter

**Files:**
- Modify: `voicebeat/app/services/segmenter.py:26-31` (SpeechRegion class)
- Modify: `voicebeat/app/services/segmenter.py:146-189` (_merge_close_words)
- Modify: `voicebeat/app/services/segmenter.py:89-99` (segment creation)

**Step 1: Add `words` to SpeechRegion**

```python
class SpeechRegion:
    """Helper class for a detected speech region."""
    def __init__(self, start: float, end: float, text: str = "", words: list[dict] | None = None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []
```

**Step 2: Preserve word timestamps in `_merge_close_words`**

The function currently accumulates `current_text` but discards individual word dicts. Add a `current_words` accumulator:

```python
def _merge_close_words(
    words: list[dict],
    gap_threshold: float,
    max_word_duration: float = 1.5,
) -> list[SpeechRegion]:
    if not words:
        return []

    regions = []
    w0 = words[0]
    current_start = w0.get("start", 0)
    current_end = min(w0.get("end", 0), current_start + max_word_duration)
    current_text = w0.get("word", "")
    current_words = [{"word": w0.get("word", ""), "start": w0.get("start", 0),
                      "end": min(w0.get("end", 0), w0.get("start", 0) + max_word_duration)}]

    for word in words[1:]:
        word_start = word.get("start", 0)
        word_end = min(word.get("end", 0), word_start + max_word_duration)
        word_text = word.get("word", "")
        word_dict = {"word": word_text, "start": word_start, "end": word_end}

        if word_start - current_end <= gap_threshold:
            current_end = word_end
            current_text += " " + word_text
            current_words.append(word_dict)
        else:
            regions.append(SpeechRegion(current_start, current_end, current_text.strip(), current_words))
            current_start = word_start
            current_end = word_end
            current_text = word_text
            current_words = [word_dict]

    regions.append(SpeechRegion(current_start, current_end, current_text.strip(), current_words))
    return regions
```

**Step 3: Pass words to AudioSegment in `segment_recording`**

At lines 91-98, update the speech segment creation:

```python
    for i, region in enumerate(speech_regions):
        segment = AudioSegment(
            id=str(uuid.uuid4()),
            type=SegmentType.SPEECH,
            start_seconds=region.start,
            end_seconds=region.end,
            transcript=region.text,
            words=region.words,
        )
```

**Step 4: Commit**

```bash
git add voicebeat/app/services/segmenter.py
git commit -m "feat: preserve word-level timestamps through segmenter pipeline"
```

---

### Task 3: Extract audio for speech segments in routes.py

**Files:**
- Modify: `voicebeat/app/api/routes.py:104-111`

Currently `routes.py` skips audio extraction for SPEECH segments (line 106: `if seg.type != SegmentType.SPEECH`). The vocal subagent needs speech audio bytes. Change the filter to extract audio for ALL segment types:

**Step 1: Update the audio extraction loop**

```python
    # Step 2: Extract audio for each segment
    logger.info("PROCESS: Step 2 - Extracting audio for each segment...")
    segment_audio_data: dict[str, bytes] = {}
    for seg in segments:
        seg_audio_path = await segmenter.extract_segment_audio(audio_bytes, seg)
        with open(seg_audio_path, "rb") as f:
            segment_audio_data[seg.id] = f.read()
        seg.audio_file = seg_audio_path
        logger.info(f"PROCESS: Extracted segment {seg.id[:8]} ({seg.type.value}) -> {seg_audio_path} ({len(segment_audio_data[seg.id])} bytes)")
```

This is the only change to `routes.py`. The rest of the pipeline (speech transcripts extraction at line 115) stays the same.

**Step 2: Commit**

```bash
git add voicebeat/app/api/routes.py
git commit -m "feat: extract audio for speech segments (needed for vocal processing)"
```

---

### Task 4: Add `pitch_shift_words()` to vocal_processor.py

**Files:**
- Modify: `voicebeat/app/services/vocal_processor.py` (add function after existing `autotune`)

**Step 1: Add the melodic mode processor**

Add this after the existing `autotune()` function (after line 236):

```python
async def pitch_shift_words(
    audio_bytes: bytes,
    words: list[dict],
    melody_design: list[dict],
    key_signature: str | None = None,
    bpm: int = 120,
) -> str:
    """
    Pitch-shift individual words to LLM-assigned MIDI notes.

    Args:
        audio_bytes: Full speech segment audio bytes
        words: Word timestamps [{word, start, end}, ...]
        melody_design: LLM output [{word, midi_note, duration_beats}, ...]
        key_signature: Key for scale reference
        bpm: Beats per minute

    Returns:
        Path to output MP3 file
    """
    sr = 22050
    spb = 60.0 / bpm  # seconds per beat

    # Load audio
    try:
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        y = y.astype(np.float32)
    except Exception:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

    # Build word-index lookup: match melody_design entries to words by text
    word_to_design = {}
    design_iter = iter(melody_design)
    for w in words:
        try:
            d = next(design_iter)
            word_to_design[id(w)] = d
        except StopIteration:
            break

    # Calculate total output length from melody design
    total_beats = 0
    for d in melody_design:
        total_beats = max(total_beats, d.get("start_beat", 0) + d.get("duration_beats", 1))
    total_seconds = total_beats * spb
    # Minimum: original audio length
    total_seconds = max(total_seconds, len(y) / sr)
    output = np.zeros(int(total_seconds * sr), dtype=np.float32)

    # Detect segment start time (first word start) for offset calculation
    seg_start = words[0]["start"] if words else 0.0

    for w in words:
        d = word_to_design.get(id(w))
        if d is None:
            continue

        target_midi = d.get("midi_note", 60)
        duration_beats = d.get("duration_beats", 1)
        start_beat = d.get("start_beat")

        # Extract word audio
        w_start = max(0, int((w["start"] - seg_start) * sr))
        w_end = min(len(y), int((w["end"] - seg_start) * sr))
        word_audio = y[w_start:w_end]
        if len(word_audio) < 512:
            continue

        # Detect current pitch of the word
        f0, voiced_flag, _ = librosa.pyin(
            word_audio, fmin=80, fmax=800, sr=sr, hop_length=512
        )
        voiced_freqs = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else np.array([])

        if len(voiced_freqs) > 0:
            avg_freq = float(np.nanmean(voiced_freqs))
            current_midi = 69 + 12 * np.log2(avg_freq / 440.0)
            shift = target_midi - current_midi
            # Clamp to +/- 7 semitones to avoid robotic artifacts
            shift = max(-7, min(7, shift))

            if abs(shift) > 0.1:
                try:
                    word_audio = librosa.effects.pitch_shift(
                        word_audio, sr=sr, n_steps=shift
                    )
                except Exception as e:
                    logger.warning(f"vocal_processor: pitch shift failed for '{w['word']}': {e}")

        # Time-stretch to match target duration if start_beat is provided
        if start_beat is not None:
            target_duration = duration_beats * spb
            current_duration = len(word_audio) / sr
            if current_duration > 0 and abs(target_duration - current_duration) > 0.05:
                rate = current_duration / target_duration
                rate = max(0.5, min(2.0, rate))  # Clamp stretch ratio
                try:
                    word_audio = librosa.effects.time_stretch(word_audio, rate=rate)
                except Exception as e:
                    logger.warning(f"vocal_processor: time stretch failed for '{w['word']}': {e}")

        # Place in output buffer
        if start_beat is not None:
            out_start = int(start_beat * spb * sr)
        else:
            out_start = int((w["start"] - seg_start) * sr)

        out_end = min(len(output), out_start + len(word_audio))
        actual_len = out_end - out_start
        if actual_len > 0:
            # Crossfade: 64 samples at boundaries
            fade = min(64, actual_len // 4)
            chunk = word_audio[:actual_len].copy()
            if fade > 0:
                chunk[:fade] *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            output[out_start:out_end] += chunk

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * (0.9 / peak)

    # Export to MP3
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"vocal_melody_{uuid.uuid4().hex[:8]}.mp3")

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, output, sr, format="WAV")
    wav_buffer.seek(0)
    audio_seg = PydubSegment.from_wav(wav_buffer)
    audio_seg.export(output_path, format="mp3")

    logger.info(f"vocal_processor: exported melodic vocal to {output_path}")
    return output_path
```

**Step 2: Commit**

```bash
git add voicebeat/app/services/vocal_processor.py
git commit -m "feat: add pitch_shift_words() for melodic vocal processing"
```

---

### Task 5: Add `beat_snap_words()` to vocal_processor.py

**Files:**
- Modify: `voicebeat/app/services/vocal_processor.py` (add function after `pitch_shift_words`)

**Step 1: Add the rhythmic mode processor**

```python
async def beat_snap_words(
    audio_bytes: bytes,
    words: list[dict],
    rhythm_design: list[dict],
    bpm: int = 120,
    total_bars: int = 4,
) -> str:
    """
    Time-stretch and reposition words onto beat grid positions.

    Args:
        audio_bytes: Full speech segment audio bytes
        words: Word timestamps [{word, start, end}, ...]
        rhythm_design: LLM output [{word, beat_position, bar}, ...]
        bpm: Beats per minute
        total_bars: Total bars for the output buffer

    Returns:
        Path to output MP3 file
    """
    sr = 22050
    spb = 60.0 / bpm  # seconds per beat
    bar_sec = 4 * spb  # seconds per bar (4/4 time)

    # Load audio
    try:
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        y = y.astype(np.float32)
    except Exception:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

    # Create output buffer
    total_seconds = total_bars * bar_sec + 1.0  # +1s padding
    output = np.zeros(int(total_seconds * sr), dtype=np.float32)

    seg_start = words[0]["start"] if words else 0.0

    # Build sorted placement list from rhythm_design
    placements = []
    design_iter = iter(rhythm_design)
    for w in words:
        try:
            d = next(design_iter)
        except StopIteration:
            break
        bar = d.get("bar", 0)
        beat_pos = d.get("beat_position", 0)  # 0-15 on 16th grid
        target_sec = bar * bar_sec + (beat_pos / 4.0) * spb
        placements.append((w, d, target_sec))

    # Sort by target time
    placements.sort(key=lambda x: x[2])

    for i, (w, d, target_sec) in enumerate(placements):
        # Extract word audio
        w_start = max(0, int((w["start"] - seg_start) * sr))
        w_end = min(len(y), int((w["end"] - seg_start) * sr))
        word_audio = y[w_start:w_end]
        if len(word_audio) < 256:
            continue

        # Calculate available slot: time until next word starts (or end of bar)
        if i + 1 < len(placements):
            slot_end = placements[i + 1][2]
        else:
            slot_end = target_sec + (w["end"] - w["start"]) * 1.5  # 1.5x original duration

        available = max(0.05, slot_end - target_sec - 0.02)  # 20ms gap between words
        current_duration = len(word_audio) / sr

        # Time-stretch if needed
        if current_duration > 0 and abs(available - current_duration) > 0.05:
            rate = current_duration / available
            rate = max(0.5, min(2.5, rate))  # Clamp stretch ratio
            try:
                word_audio = librosa.effects.time_stretch(word_audio, rate=rate)
            except Exception as e:
                logger.warning(f"vocal_processor: time stretch failed for '{w['word']}': {e}")

        # Place in output buffer
        out_start = int(target_sec * sr)
        out_end = min(len(output), out_start + len(word_audio))
        actual_len = out_end - out_start
        if actual_len > 0:
            fade = min(64, actual_len // 4)
            chunk = word_audio[:actual_len].copy()
            if fade > 0:
                chunk[:fade] *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            output[out_start:out_end] += chunk

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * (0.9 / peak)

    # Export to MP3
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"vocal_rhythm_{uuid.uuid4().hex[:8]}.mp3")

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, output, sr, format="WAV")
    wav_buffer.seek(0)
    audio_seg = PydubSegment.from_wav(wav_buffer)
    audio_seg.export(output_path, format="mp3")

    logger.info(f"vocal_processor: exported rhythmic vocal to {output_path}")
    return output_path
```

**Step 2: Commit**

```bash
git add voicebeat/app/services/vocal_processor.py
git commit -m "feat: add beat_snap_words() for rhythmic vocal processing"
```

---

### Task 6: Rewrite vocal subagent as LLM-backed agent

**Files:**
- Rewrite: `voicebeat/app/services/subagents/vocal.py`

**Step 1: Replace the entire file**

The new vocal subagent inherits `BaseSubagent`, is parameterized by `mode` ("melodic" or "rhythmic"), and exposes mode-specific tools. It stores the LLM's design in `self._design` and renders on `render_vocal`.

```python
"""
Vocal Subagent — LLM-designed vocal processing.

Melodic mode: LLM assigns MIDI notes to words, processor pitch-shifts.
Rhythmic mode: LLM assigns beat positions to words, processor time-snaps.
"""

import json
import uuid
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.models.schemas import Layer, Project, Instrument, SegmentType
from app.services import vocal_processor
from app.services.subagents.base import BaseSubagent

logger = logging.getLogger(__name__)


# ── Tool definitions ──────────────────────────────────────────────────

TOOL_DESIGN_MELODY = {
    "type": "function",
    "function": {
        "name": "design_vocal_melody",
        "description": (
            "Assign a MIDI note and timing to each word to create a vocal melody. "
            "The user's speech will be pitch-shifted to match these notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "description": "The word text"},
                            "midi_note": {
                                "type": "integer",
                                "description": "Target MIDI note (60=C4, 62=D4, 64=E4, etc.)",
                            },
                            "start_beat": {
                                "type": "number",
                                "description": "Beat position to start this word (0-based, quarter note beats)",
                            },
                            "duration_beats": {
                                "type": "number",
                                "description": "How long the word should last in beats",
                            },
                        },
                        "required": ["word", "midi_note", "start_beat", "duration_beats"],
                    },
                },
            },
            "required": ["notes"],
        },
    },
}

TOOL_DESIGN_RHYTHM = {
    "type": "function",
    "function": {
        "name": "design_vocal_rhythm",
        "description": (
            "Assign beat-grid positions to each word for rhythmic vocal placement. "
            "The user's speech will be time-stretched and repositioned to snap to these positions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "placements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "description": "The word text"},
                            "bar": {"type": "integer", "description": "Which bar (0-indexed)"},
                            "beat_position": {
                                "type": "integer",
                                "description": "Position on 16th-note grid (0-15) within the bar",
                            },
                        },
                        "required": ["word", "bar", "beat_position"],
                    },
                },
            },
            "required": ["placements"],
        },
    },
}

TOOL_RENDER_VOCAL = {
    "type": "function",
    "function": {
        "name": "render_vocal",
        "description": "Render the designed vocal — applies pitch-shifts or time-stretches to produce audio.",
        "parameters": {
            "type": "object",
            "properties": {
                "layer_name": {
                    "type": "string",
                    "description": "Name for the vocal layer (e.g. 'Lead Vocals', 'Rap Flow')",
                },
            },
            "required": ["layer_name"],
        },
    },
}

TOOL_DONE = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "Signal that vocal processing is complete.",
        "parameters": {"type": "object", "properties": {}},
    },
}


# ── Vocal Subagent ────────────────────────────────────────────────────

class VocalSubagent(BaseSubagent):
    """LLM-backed vocal processor with melodic and rhythmic modes."""

    name = "vocal"
    max_iterations = 4

    def __init__(self, mode: str = "melodic"):
        self.mode = mode  # "melodic" or "rhythmic"
        self._design: list[dict] | None = None

    def get_system_prompt(self, project: Project, instructions: dict) -> str:
        key = project.key_signature or "C major"
        bpm = project.bpm

        if self.mode == "melodic":
            return f"""You are a vocal melody designer. You receive spoken lyrics with word timestamps
and design a singing melody for them.

## Your task
Assign a MIDI note, start beat, and duration to each word using design_vocal_melody.
Then call render_vocal to produce the audio. Then call done.

## Musical context
- Key: {key}
- BPM: {bpm}
- Scale notes for {key}: use notes that fit this key

## Guidelines
- Keep the melody simple and singable — stepwise motion, small intervals
- Match natural speech emphasis: stressed syllables get higher notes or longer durations
- Short words (articles, prepositions) can share a note with adjacent words
- Typical vocal range: MIDI 55-72 (G3 to C5)
- Place words on or near beat boundaries for a natural feel
- Leave small gaps between phrases for breathing"""

        else:  # rhythmic
            return f"""You are a vocal rhythm designer for rap/spoken word. You receive spoken lyrics
with word timestamps and design beat-grid placements for them.

## Your task
Assign a bar and beat_position (0-15 on the 16th-note grid) to each word using design_vocal_rhythm.
Then call render_vocal to produce the audio. Then call done.

## Musical context
- BPM: {bpm}
- 16th-note grid: positions 0-15 per bar, where 0=downbeat, 4=beat 2, 8=beat 3, 12=beat 4

## Guidelines
- Emphasize the groove — key words on strong beats (0, 4, 8, 12)
- Pack syllables tightly for fast flow, space them for emphasis
- Leave position 0 of bar 0 for a strong opening word
- End phrases before the next bar for breathing room
- Rhyming words should land on similar beat positions for rhythmic consistency"""

    def get_tools(self) -> list[dict]:
        if self.mode == "melodic":
            return [TOOL_DESIGN_MELODY, TOOL_RENDER_VOCAL, TOOL_DONE]
        else:
            return [TOOL_DESIGN_RHYTHM, TOOL_RENDER_VOCAL, TOOL_DONE]

    def build_user_message(self, project: Project, instructions: dict) -> str:
        words = instructions.get("words", [])
        lines = [f"# Vocal Assignment\n"]
        lines.append(f"- BPM: {project.bpm}")
        lines.append(f"- Key: {project.key_signature or 'C major'}")
        lines.append(f"- Genre: {instructions.get('genre', 'pop')}")
        lines.append(f"- Mode: {self.mode}\n")

        if instructions.get("style_notes"):
            lines.append(f"## Style: {instructions['style_notes']}\n")

        lines.append("## Lyrics with word timestamps:")
        for i, w in enumerate(words, 1):
            dur = w["end"] - w["start"]
            lines.append(f'{i}. "{w["word"]}" — {w["start"]:.2f}s–{w["end"]:.2f}s ({dur:.2f}s)')

        if instructions.get("sections"):
            total_bars = max((s.get("end_bar", 0) for s in instructions["sections"]), default=4)
            lines.append(f"\n## Target: fit lyrics into {total_bars} bars")

        lines.append(f"\nDesign the vocal {'melody' if self.mode == 'melodic' else 'rhythm'} now.")
        lines.append("Call the design tool, then render_vocal, then done.")
        return "\n".join(lines)

    async def execute_tool(
        self,
        tool_call,
        project: Project,
        instructions: dict,
        created_layers: list[Layer],
    ) -> dict:
        fname = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if fname == "design_vocal_melody":
            self._design = args.get("notes", [])
            return {
                "success": True,
                "message": f"Melody designed for {len(self._design)} words",
            }

        elif fname == "design_vocal_rhythm":
            self._design = args.get("placements", [])
            return {
                "success": True,
                "message": f"Rhythm designed for {len(self._design)} words",
            }

        elif fname == "render_vocal":
            if self._design is None:
                return {"success": False, "error": "No design yet — call design tool first"}

            layer_name = args.get("layer_name", "Vocals")
            words = instructions.get("words", [])
            segment_audio = instructions.get("segment_audio", {})
            segment_id = instructions.get("segment_id")

            # Resolve audio bytes
            audio_bytes = None
            if segment_id:
                audio_bytes = segment_audio.get(segment_id)
                if audio_bytes is None:
                    for sid, data in segment_audio.items():
                        if sid.startswith(segment_id) or segment_id.startswith(sid):
                            audio_bytes = data
                            break

            if audio_bytes is None:
                return {"success": False, "error": f"No audio found for segment {segment_id}"}

            try:
                if self.mode == "melodic":
                    audio_file = await vocal_processor.pitch_shift_words(
                        audio_bytes=audio_bytes,
                        words=words,
                        melody_design=self._design,
                        key_signature=project.key_signature,
                        bpm=project.bpm,
                    )
                else:
                    total_bars = 4
                    if instructions.get("sections"):
                        total_bars = max(
                            (s.get("end_bar", 0) for s in instructions["sections"]),
                            default=4,
                        )
                    audio_file = await vocal_processor.beat_snap_words(
                        audio_bytes=audio_bytes,
                        words=words,
                        rhythm_design=self._design,
                        bpm=project.bpm,
                        total_bars=total_bars,
                    )
            except Exception as e:
                logger.error(f"VOCAL: Render failed: {e}")
                return {"success": False, "error": f"Render failed: {e}"}

            layer = Layer(
                id=str(uuid.uuid4()),
                name=layer_name,
                instrument=Instrument.VOCAL,
                segment_type=SegmentType.VOCAL,
                audio_file=audio_file,
            )
            created_layers.append(layer)

            return {
                "success": True,
                "message": f"Vocal rendered: '{layer_name}' -> {audio_file}",
                "layer_id": layer.id,
            }

        return {"success": False, "error": f"Unknown tool: {fname}"}
```

**Step 2: Update `subagents/__init__.py`**

No change needed — `VocalSubagent` is already exported. The class name stays the same, just the implementation changes.

**Step 3: Commit**

```bash
git add voicebeat/app/services/subagents/vocal.py
git commit -m "feat: rewrite vocal subagent as LLM-backed agent with melodic/rhythmic modes"
```

---

### Task 7: Update director to support speech-to-vocal delegation

**Files:**
- Modify: `voicebeat/app/services/director.py`

This task has multiple sub-changes in director.py.

**Step 1: Update `delegate_vocal` tool definition (lines 160-180)**

Add `mode` parameter and update the description:

```python
    {
        "type": "function",
        "function": {
            "name": "delegate_vocal",
            "description": (
                "Create vocals from a user's speech or singing segment. "
                "For speech segments: the vocal subagent designs a melody or rhythm for the lyrics. "
                "For melody segments: applies autotune to existing singing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "The segment ID containing speech lyrics or singing",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["melodic", "rhythmic"],
                        "description": "Vocal style: 'melodic' for singing (pop/R&B/jazz), 'rhythmic' for rap/spoken word (hip-hop/electronic)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Layer name (e.g. 'Lead Vocals', 'Rap Flow')",
                    },
                },
                "required": ["segment_id", "mode"],
            },
        },
    },
```

**Step 2: Update `_handle_phase1_tool` for delegate_vocal (around line 448)**

Replace the existing `delegate_vocal` handler. The new version resolves the segment, detects whether it's speech (needs LLM vocal design) or melody (autotune path), and passes word timestamps through:

```python
        elif name == "delegate_vocal":
            sid = args["segment_id"]
            mode = args.get("mode", "melodic")

            # Resolve segment ID in audio data
            resolved_sid = None
            if sid in segment_audio:
                resolved_sid = sid
            else:
                for full_id in segment_audio:
                    if full_id.startswith(sid) or sid.startswith(full_id):
                        resolved_sid = full_id
                        break

            if resolved_sid is None:
                available = list(segment_audio.keys())
                if available:
                    short_ids = [s[:8] for s in available]
                    return {
                        "success": False,
                        "error": f"Segment '{sid}' not found. Available: {short_ids}",
                    }
                else:
                    return {
                        "success": False,
                        "error": "No audio segments available. Cannot create vocals without audio.",
                    }

            # Find the AudioSegment object to get word timestamps
            words = None
            for seg in self._current_segments:
                if seg.id == resolved_sid or seg.id.startswith(sid) or sid.startswith(seg.id):
                    words = seg.words
                    break

            d = {
                "type": "vocal",
                "segment_id": resolved_sid,
                "mode": mode,
                "name": args.get("name", "Vocals"),
                "words": words or [],
                "genre": song_plan.get("genre", "pop"),
                "style_notes": args.get("style_notes", ""),
                "sections": song_plan.get("sections", []),
                "segment_audio": segment_audio,
            }
            delegations.append(d)
            word_count = len(words) if words else 0
            return {
                "success": True,
                "message": f"Vocal delegated: {mode} mode, segment {resolved_sid[:8]}, {word_count} words",
            }
```

**Step 3: Store segments on the Director for word lookup**

In the `run()` method, store the segments list so `_handle_phase1_tool` can access it:

At the start of `run()`, after the first log line, add:
```python
        self._current_segments = segments
```

Similarly in `run_refinement()`:
```python
        self._current_segments = segments or []
```

**Step 4: Update `_run_subagents` to pass `mode` to VocalSubagent**

In `_run_subagents`, update the vocal branch (around line 574):

```python
            elif d["type"] == "vocal":
                agent = VocalSubagent(mode=d.get("mode", "melodic"))
                tasks.append(agent.run(project, d))
```

**Step 5: Update context builders for speech-as-vocal**

In `_build_context()`, replace the existing vocal candidates section (around line 903) with one that includes both melody segments AND speech segments with lyrics:

```python
        # List segments available for delegate_vocal
        vocal_candidates = []
        for segment in segments:
            if segment.type == SegmentType.SPEECH and segment.words and len(segment.words) > 2:
                word_count = len(segment.words)
                vocal_candidates.append(
                    f"- {segment.id} (speech, {word_count} words: \"{segment.transcript[:60]}...\")"
                    if len(segment.transcript or "") > 60
                    else f"- {segment.id} (speech, {word_count} words: \"{segment.transcript}\")"
                )
            elif segment.type == SegmentType.MELODY:
                vocal_candidates.append(f"- {segment.id} (singing/melody)")

        if vocal_candidates:
            lines.append("## Segments available for delegate_vocal:")
            lines.append("Use mode='melodic' for singing, mode='rhythmic' for rap/spoken word.")
            lines.extend(vocal_candidates)
            lines.append("")
        else:
            lines.append("## Vocals: No segments available for vocals.\n")
```

Apply the same pattern to `_build_refinement_context()`.

**Step 6: Update Phase 1 system prompt vocal section**

Replace the existing `## Vocals` section with:

```
## Vocals
- delegate_vocal works with BOTH speech segments (spoken lyrics) and melody segments (singing)
- For speech segments: set mode='melodic' (pop/R&B/jazz/lo-fi) or mode='rhythmic' (hip-hop/electronic/rock)
- For melody segments: either mode works (autotune is applied to existing singing)
- Speech segments with >2 words can be vocalized — the vocal subagent designs the melody/rhythm
- If the user asks for vocals and has a speech segment with lyrics, USE IT
```

**Step 7: Commit**

```bash
git add voicebeat/app/services/director.py
git commit -m "feat: update director to support speech-to-vocal delegation with mode selection"
```

---

### Task 8: Verify end-to-end

**Step 1: Start the server**

```bash
cd voicebeat && python -m app.main
```

**Step 2: Test with a recording**

Record audio that includes spoken lyrics (e.g., "make me a hip-hop beat... I wanna fly above the clouds tonight") and send it through the `/api/v1/process` endpoint.

**Step 3: Verify in logs**

Check for:
- `SEGMENTER: Speech segment` entries with word counts
- `DIRECTOR P1: delegate_vocal` with mode and word count
- `VOCAL: Iteration 1` (LLM loop running)
- `VOCAL: Tool call: design_vocal_rhythm(...)` or `design_vocal_melody(...)`
- `VOCAL: Tool call: render_vocal(...)`
- `vocal_processor: exported` (audio file created)
- The final mix includes the vocal layer

**Step 4: Listen to output**

Play the mixed file and verify the vocal track has audible, musically-placed words.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: speech-to-musical-vocals pipeline complete"
```
