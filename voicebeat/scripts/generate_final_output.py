"""
Generate the final sample-output.m4a file.

Usage:
    cd voicebeat && python scripts/generate_final_output.py
"""

import asyncio
import sys
from pathlib import Path

# Allow imports from the voicebeat root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydub import AudioSegment as PydubSegment

from app.services import pitch_analyzer, composium_bridge
from config.settings import settings


def convert_to_wav_bytes(path: str) -> bytes:
    """Convert any audio file to WAV bytes using pydub."""
    import io
    audio = PydubSegment.from_file(path)
    audio = audio.set_frame_rate(22050).set_channels(1)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()


async def main():
    samples_dir = Path(__file__).parent.parent / "samples"
    input_file = samples_dir / "sample-input.m4a"
    output_file = samples_dir / "sample-output.m4a"

    print("=" * 70)
    print("GENERATING FINAL OUTPUT")
    print("=" * 70)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return

    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")

    print(f"\nCurrent settings:")
    print(f"  pitch_fmin: {settings.pitch_fmin}")
    print(f"  pitch_fmax: {settings.pitch_fmax}")
    print(f"  min_note_duration: {settings.min_note_duration}")
    print(f"  onset_delta: {settings.onset_delta}")
    print(f"  pitch_shift: {settings.pitch_shift}")

    # Load input as WAV bytes
    print("\nConverting input to WAV...")
    input_bytes = convert_to_wav_bytes(str(input_file))

    # Run melody analysis
    print("Analyzing melody...")
    melody = await pitch_analyzer.analyze_melody(input_bytes)

    print(f"\nExtracted melody:")
    print(f"  Notes detected: {len(melody.pitches)}")
    print(f"  Key signature: {melody.key_signature}")
    print(f"  Tempo: {melody.tempo_bpm} BPM")

    if melody.pitches:
        midi_notes = [p.midi_note for p in melody.pitches]
        print(f"  MIDI range: {min(midi_notes)} - {max(midi_notes)}")

    # Render the melody with guitar
    print("\nRendering melody with guitar sound...")
    mp3_output = await composium_bridge.render_melody(
        melody,
        instrument="guitar",
        bpm=melody.tempo_bpm or 120,
        key=melody.key_signature,
    )
    print(f"Rendered MP3: {mp3_output}")

    # Convert to m4a
    print("\nConverting to M4A format...")
    audio = PydubSegment.from_file(mp3_output)
    audio.export(str(output_file), format="ipod")  # ipod format = m4a/aac
    print(f"Exported M4A: {output_file}")

    # Verify the output
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"\nSuccess! Output file created:")
        print(f"  Path: {output_file}")
        print(f"  Size: {size:,} bytes")
    else:
        print("\nERROR: Output file was not created!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
