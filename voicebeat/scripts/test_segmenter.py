"""
Test script for the hybrid segmenter.

Usage:
    python scripts/test_segmenter.py <audio_file>

Example:
    python scripts/test_segmenter.py /path/to/recording.wav

Prints each detected segment with all its attributes.
"""

import asyncio
import sys
from pathlib import Path

# Allow imports from the voicebeat root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import segmenter
from app.utils.audio import get_content_type


async def main(audio_path: str) -> None:
    path = Path(audio_path)
    if not path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Testing segmenter on: {path.name}")
    print(f"{'='*60}\n")

    audio_bytes = path.read_bytes()
    content_type = get_content_type(path.name)
    print(f"File size : {len(audio_bytes):,} bytes")
    print(f"MIME type : {content_type}\n")

    print("Running segmenter...\n")
    segments = await segmenter.segment_recording(audio_bytes, content_type)

    print(f"{'='*60}")
    print(f"  {len(segments)} segment(s) found")
    print(f"{'='*60}\n")

    for i, seg in enumerate(segments, 1):
        duration = seg.end_seconds - seg.start_seconds
        print(f"  Segment {i}")
        print(f"    id               : {seg.id}")
        print(f"    type             : {seg.type.value}")
        print(f"    start            : {seg.start_seconds:.3f}s")
        print(f"    end              : {seg.end_seconds:.3f}s")
        print(f"    duration         : {duration:.3f}s")
        print(f"    instrument       : {seg.instrument.value if seg.instrument else None}")
        print(f"    semantic_command : {seg.semantic_command!r}")
        print(f"    transcript       : {seg.transcript!r}")
        print(f"    volume           : {seg.volume:.3f}" if seg.volume is not None else "    volume           : None")
        print(f"    audio_file       : {seg.audio_file}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_segmenter.py <audio_file>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
