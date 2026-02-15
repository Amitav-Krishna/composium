"""
Vocal Processor Service

Autotune pipeline: pitch-corrects user vocals to the nearest scale note
using librosa. Preserves the original voice timbre while snapping pitches
to musically correct frequencies.
"""

import asyncio
import io
import uuid
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment as PydubSegment
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from app.services import tts

logger = logging.getLogger(__name__)

# Chromatic scale frequencies are derived from A4=440 Hz
# Scale intervals (semitone offsets from root)
_SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}

_NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def _parse_key_signature(key_signature: str | None) -> tuple[int, str]:
    """Parse key signature string like 'C major' or 'A minor' into (root_semitone, mode)."""
    if not key_signature:
        return 0, "major"  # Default to C major

    parts = key_signature.strip().split()
    root_name = parts[0] if parts else "C"
    mode = parts[1].lower() if len(parts) > 1 else "major"

    root = _NOTE_TO_SEMITONE.get(root_name, 0)

    if mode.startswith("min"):
        mode = "minor"
    else:
        mode = "major"

    return root, mode


def _build_scale_midi_notes(root_semitone: int, mode: str) -> set[int]:
    """Build a set of all MIDI note numbers that belong to the given scale."""
    intervals = _SCALE_INTERVALS.get(mode, _SCALE_INTERVALS["major"])
    scale_pcs = set((root_semitone + i) % 12 for i in intervals)
    # All MIDI notes in range 20-108 that belong to this scale
    return {m for m in range(20, 109) if m % 12 in scale_pcs}


def _nearest_scale_midi(midi_note: float, scale_notes: set[int]) -> int:
    """Find nearest MIDI note in the scale to the given (possibly fractional) MIDI pitch."""
    rounded = round(midi_note)
    best = None
    best_dist = float('inf')
    # Search within +/- 6 semitones for nearest scale note
    for offset in range(-6, 7):
        candidate = rounded + offset
        if candidate in scale_notes:
            dist = abs(midi_note - candidate)
            if dist < best_dist:
                best = candidate
                best_dist = dist
    return best if best is not None else rounded


async def autotune(
    audio_bytes: bytes,
    key_signature: str | None = None,
    strength: float = 0.8,
) -> str:
    """
    Apply autotune pitch correction to vocal audio.

    Args:
        audio_bytes: Raw audio bytes of the vocal recording
        key_signature: Key to snap to (e.g. "C major"). Auto-detected if None.
        strength: How aggressively to correct (0.0 = none, 1.0 = full snap)

    Returns:
        Path to the output MP3 file
    """
    sr = 22050

    # 1. Load audio
    try:
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        y = y.astype(np.float32)
    except Exception:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

    # 2. Detect pitch contour
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=80, fmax=800, sr=sr, hop_length=512
    )

    hop_length = 512

    # 3. Detect key from pitch classes if not provided
    if not key_signature:
        key_signature = _detect_key_from_f0(f0, voiced_flag)

    root_semitone, mode = _parse_key_signature(key_signature)
    scale_notes = _build_scale_midi_notes(root_semitone, mode)

    logger.info(f"vocal_processor: autotuning in {key_signature}, strength={strength}")

    # 4. Group consecutive voiced frames into segments and pitch-correct
    output = np.copy(y)
    n_frames = len(f0)
    i = 0

    while i < n_frames:
        # Skip unvoiced frames
        if not voiced_flag[i] or np.isnan(f0[i]):
            i += 1
            continue

        # Find contiguous voiced segment
        seg_start = i
        while i < n_frames and voiced_flag[i] and not np.isnan(f0[i]):
            i += 1
        seg_end = i

        if seg_end - seg_start < 2:
            continue

        # Calculate average frequency for this segment
        seg_freqs = f0[seg_start:seg_end]
        avg_freq = np.nanmean(seg_freqs)
        if avg_freq <= 0 or np.isnan(avg_freq):
            continue

        # Convert to MIDI
        midi_note = 69 + 12 * np.log2(avg_freq / 440.0)

        # Find nearest scale note
        target_midi = _nearest_scale_midi(midi_note, scale_notes)

        # Calculate semitone shift
        shift = target_midi - midi_note

        # Apply strength
        shift *= strength

        # Only correct if shift is significant (> 0.1 semitones)
        if abs(shift) < 0.1:
            continue

        # Get audio sample range for this segment (with small padding for crossfade)
        sample_start = max(0, seg_start * hop_length - hop_length)
        sample_end = min(len(y), seg_end * hop_length + hop_length)
        segment_audio = y[sample_start:sample_end]

        if len(segment_audio) < 512:
            continue

        # Apply pitch shift
        try:
            shifted = librosa.effects.pitch_shift(
                segment_audio, sr=sr, n_steps=shift
            )

            # Crossfade edges (32 samples)
            fade_len = min(32, len(shifted) // 4)
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)

                # Blend start
                inner_start = seg_start * hop_length
                inner_end = seg_end * hop_length
                inner_start = max(0, inner_start)
                inner_end = min(len(y), inner_end)

                shifted_inner = shifted[
                    inner_start - sample_start : inner_end - sample_start
                ]
                if len(shifted_inner) > 0:
                    # Apply fade at boundaries
                    if len(shifted_inner) > fade_len * 2:
                        shifted_inner[:fade_len] *= fade_in
                        shifted_inner[-fade_len:] *= fade_out
                        # Blend with original at edges
                        orig_slice = output[inner_start:inner_end]
                        if len(orig_slice) == len(shifted_inner):
                            blend = np.copy(shifted_inner)
                            blend[:fade_len] += orig_slice[:fade_len] * fade_out
                            blend[-fade_len:] += orig_slice[-fade_len:] * fade_in
                            output[inner_start:inner_end] = blend
                        else:
                            output[inner_start:inner_start + len(shifted_inner)] = shifted_inner
                    else:
                        output[inner_start:inner_start + len(shifted_inner)] = shifted_inner
        except Exception as e:
            logger.warning(f"vocal_processor: pitch shift failed for segment: {e}")
            continue

    # 5. Export to MP3
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"vocal_{uuid.uuid4().hex[:8]}.mp3")

    # Write WAV to buffer, then convert to MP3 via pydub
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, output, sr, format="WAV")
    wav_buffer.seek(0)

    audio_seg = PydubSegment.from_wav(wav_buffer)
    audio_seg.export(output_path, format="mp3")

    logger.info(f"vocal_processor: exported autotuned vocal to {output_path}")
    return output_path


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


def _detect_key_from_f0(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
) -> str:
    """Detect key signature from pitch contour using pitch class histogram."""
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    pitch_class_counts = np.zeros(12)
    for freq, voiced in zip(f0, voiced_flag):
        if voiced and not np.isnan(freq) and freq > 0:
            midi = 69 + 12 * np.log2(freq / 440.0)
            pc = round(midi) % 12
            pitch_class_counts[pc] += 1

    total = np.sum(pitch_class_counts)
    if total == 0:
        return "C major"
    pitch_class_counts /= total

    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    major_template /= major_template.sum()
    minor_template /= minor_template.sum()

    best_key = "C major"
    best_score = -1.0

    for root in range(12):
        major_score = float(np.dot(pitch_class_counts, np.roll(major_template, root)))
        minor_score = float(np.dot(pitch_class_counts, np.roll(minor_template, root)))

        if major_score > best_score:
            best_score = major_score
            best_key = f"{NOTE_NAMES[root]} major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{NOTE_NAMES[root]} minor"

    return best_key


async def tts_melodic_vocal(
    melody_design: list[dict],
    bpm: int = 120,
    voice_id: str | None = None,
) -> str:
    """
    Generate melodic vocals from TTS — pitch-shift and time-stretch each word
    to match the LLM-designed melody.

    Args:
        melody_design: [{word, midi_note, start_beat, duration_beats}, ...]
        bpm: Beats per minute
        voice_id: TTS voice ID override

    Returns:
        Path to output MP3 file
    """
    sr = 22050
    spb = 60.0 / bpm

    logger.info(f"vocal_processor: TTS generating {len(melody_design)} words (melodic)")

    # 1. Generate TTS for all words in parallel
    tts_results = await asyncio.gather(
        *[tts.speak_word(d["word"], voice_id) for d in melody_design]
    )

    # 2. Calculate output buffer length
    total_beats = 0
    for d in melody_design:
        total_beats = max(total_beats, d.get("start_beat", 0) + d.get("duration_beats", 1))
    total_seconds = total_beats * spb + 1.0  # +1s padding
    output = np.zeros(int(total_seconds * sr), dtype=np.float32)

    # 3. Process each word
    for d, wav_bytes in zip(melody_design, tts_results):
        start_beat = d.get("start_beat", 0)

        # Load TTS WAV
        try:
            y, orig_sr = sf.read(io.BytesIO(wav_bytes))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
        except Exception as e:
            logger.warning(f"vocal_processor: failed to load TTS for '{d['word']}': {e}")
            continue

        # Resample from TTS sample rate to working rate
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) > 256:
            y = y_trimmed

        # Place in output buffer with crossfade
        out_start = int(start_beat * spb * sr)
        out_end = min(len(output), out_start + len(y))
        actual_len = out_end - out_start
        if actual_len > 0:
            fade = min(64, actual_len // 4)
            chunk = y[:actual_len].copy()
            if fade > 0:
                chunk[:fade] *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            output[out_start:out_end] += chunk

    # 4. Normalize and export
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * (0.9 / peak)

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


async def tts_rhythmic_vocal(
    rhythm_design: list[dict],
    bpm: int = 120,
    total_bars: int = 4,
    voice_id: str | None = None,
) -> str:
    """
    Generate rhythmic vocals from TTS — time-stretch words onto beat grid.
    No pitch-shifting (rap/spoken word mode).

    Args:
        rhythm_design: [{word, bar, beat_position}, ...]
        bpm: Beats per minute
        total_bars: Total bars for the output buffer
        voice_id: TTS voice ID override

    Returns:
        Path to output MP3 file
    """
    sr = 22050
    spb = 60.0 / bpm
    bar_sec = 4 * spb

    logger.info(f"vocal_processor: TTS generating {len(rhythm_design)} words (rhythmic)")

    # 1. Generate TTS for all words in parallel
    tts_results = await asyncio.gather(
        *[tts.speak_word(d["word"], voice_id) for d in rhythm_design]
    )

    # 2. Create output buffer
    total_seconds = total_bars * bar_sec + 1.0
    output = np.zeros(int(total_seconds * sr), dtype=np.float32)

    # 3. Build sorted placement list
    placements = []
    for d, wav_bytes in zip(rhythm_design, tts_results):
        bar = d.get("bar", 0)
        beat_pos = d.get("beat_position", 0)
        target_sec = bar * bar_sec + (beat_pos / 4.0) * spb
        placements.append((d, wav_bytes, target_sec))

    placements.sort(key=lambda x: x[2])

    # 4. Process each word
    for i, (d, wav_bytes, target_sec) in enumerate(placements):
        # Load TTS WAV
        try:
            y, orig_sr = sf.read(io.BytesIO(wav_bytes))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
        except Exception as e:
            logger.warning(f"vocal_processor: failed to load TTS for '{d['word']}': {e}")
            continue

        # Resample
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) > 256:
            y = y_trimmed

        # Place in output buffer
        out_start = int(target_sec * sr)
        out_end = min(len(output), out_start + len(y))
        actual_len = out_end - out_start
        if actual_len > 0:
            fade = min(64, actual_len // 4)
            chunk = y[:actual_len].copy()
            if fade > 0:
                chunk[:fade] *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            output[out_start:out_end] += chunk

    # 5. Normalize and export
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * (0.9 / peak)

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
