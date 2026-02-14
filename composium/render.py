"""Rendering pipeline — ABC → MIDI → MP3."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

from composium.notation import Score, score_to_abc


def render(score: Score, output_path: str, keep_intermediates: bool = True) -> None:
    """Render a Score to an MP3 file via ABC → MIDI → MP3 pipeline."""
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
    subprocess.run(
        f"timidity {mid_path} -Ow -o - | ffmpeg -y -i - {trim} {output_path}",
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
