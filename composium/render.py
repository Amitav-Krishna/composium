"""Rendering pipeline — ABC → MIDI → MP3."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

from composium.notation import Score, score_to_abc


def render(
    score: Score,
    output_path: str,
    keep_intermediates: bool = True,
    pitch_shift: int = 0,
) -> None:
    """Render a Score to an MP3 file via ABC → MIDI → MP3 pipeline.

    Args:
        score: The Score to render
        output_path: Path to save the output MP3
        keep_intermediates: Whether to keep ABC and MIDI files
        pitch_shift: Semitones to shift pitch (positive=higher, negative=lower)
    """
    out = Path(output_path)
    stem = out.with_suffix("")
    abc_path = str(stem) + ".abc"
    mid_path = str(stem) + ".mid"

    # 1. Generate ABC notation
    abc_text = score_to_abc(score)
    with open(abc_path, "w") as f:
        f.write(abc_text)

    # 2. ABC → MIDI
    subprocess.run(
        ["abc2midi", abc_path, "-o", mid_path],
        check=True,
        capture_output=True,
    )

    # 3. MIDI → MP3 (timidity renders to WAV, piped to ffmpeg for MP3)
    # Trim to score duration to remove timidity's sustain/reverb tail
    trim = f"-t {score.duration:.2f}" if score.duration > 0 else ""

    # Build pitch shift filter if needed
    # Using asetrate to change pitch and aresample to restore sample rate
    # This shifts pitch without changing tempo
    pitch_filter = ""
    if pitch_shift != 0:
        # Calculate the rate multiplier: 2^(semitones/12)
        rate_mult = 2 ** (pitch_shift / 12)
        # asetrate changes the sample rate (and thus pitch), then aresample restores it
        pitch_filter = f"-af asetrate=44100*{rate_mult},aresample=44100"

    subprocess.run(
        f"timidity {mid_path} -Ow -o - | ffmpeg -y -i - {pitch_filter} {trim} {output_path}",
        shell=True,
        check=True,
        capture_output=True,
    )

    # 4. Verify output duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        actual_duration = float(result.stdout.strip())
        if score.duration > 0 and abs(actual_duration - score.duration) > 2.0:
            print(
                f"Warning: output duration ({actual_duration:.1f}s) differs from "
                f"score duration ({score.duration:.1f}s) by more than 2s"
            )

    # 5. Cleanup intermediates if requested
    if not keep_intermediates:
        for path in (abc_path, mid_path):
            if os.path.exists(path):
                os.remove(path)


def _build_volume_expr(envelope: list[tuple[float, float]]) -> str:
    """Build an ffmpeg volume expression for piecewise-linear interpolation.

    *envelope* is a sorted list of ``(time_sec, volume_0_to_1)`` breakpoints.
    Between breakpoints the volume is linearly interpolated.
    """
    if not envelope:
        return "1"
    if len(envelope) == 1:
        return str(envelope[0][1])

    # Build nested if(lt(t,...), lerp, ...) expression from right to left.
    # After the last breakpoint, hold the final volume.
    expr = str(envelope[-1][1])

    for i in range(len(envelope) - 2, -1, -1):
        t0, v0 = envelope[i]
        t1, v1 = envelope[i + 1]
        dt = t1 - t0
        if dt < 0.001:
            # Essentially a step — use the earlier value before this time
            segment = str(v0)
        else:
            # Linear interpolation: v0 + (v1-v0) * (t - t0) / (t1 - t0)
            segment = f"{v0}+{v1 - v0}*(t-{t0})/{dt}"
        expr = f"if(lt(t,{t1}),{segment},{expr})"

    return expr


def layer_with_envelope(
    base_audio: str,
    overlay_audio: str,
    output_path: str,
    envelope: list[tuple[float, float]],
    duration: float | None = None,
) -> None:
    """Mix *base_audio* with *overlay_audio* whose volume follows *envelope*.

    *envelope* is a list of ``(time_sec, volume_0_to_1)`` breakpoints.
    The overlay's volume is piecewise-linearly interpolated between them.
    """
    vol_expr = _build_volume_expr(envelope)
    filter_str = (
        f"[1:a]volume='{vol_expr}':eval=frame[voc];"
        f"[0:a][voc]amix=inputs=2:duration=longest:normalize=0"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", base_audio,
        "-i", overlay_audio,
        "-filter_complex", filter_str,
    ]
    if duration is not None:
        cmd.extend(["-t", f"{duration:.2f}"])
    cmd.append(output_path)

    subprocess.run(cmd, check=True, capture_output=True)


def layer(
    audio_paths: Sequence[str],
    output_path: str,
    duration: float | None = None,
) -> None:
    """Mix multiple audio files together into a single output using ffmpeg.

    All inputs are overlaid from t=0.  The output length matches the longest
    input unless *duration* is given.
    """
    if not audio_paths:
        raise ValueError("No audio files to layer")

    inputs: list[str] = []
    for path in audio_paths:
        inputs.extend(["-i", path])

    n = len(audio_paths)
    filter_str = f"amix=inputs={n}:duration=longest:normalize=0"

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_str]
    if duration is not None:
        cmd.extend(["-t", f"{duration:.2f}"])
    cmd.append(output_path)

    subprocess.run(cmd, check=True, capture_output=True)
