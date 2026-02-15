import sys
import tempfile
import uuid
from pathlib import Path

from pydub import AudioSegment

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from config.settings import settings

from app.models.schemas import Project, RhythmPattern
from app.services.file_storage import FileStorageService


async def render_layer(
    rhythm: RhythmPattern,
    sample_mapping: dict[str, str],
    storage: FileStorageService,
    bpm: int | None = None,
    output_dir: Path | None = None,
) -> str:
    """
    DEPRECATED: Use composium_bridge.render_rhythm() instead, which renders
    via Composium's MIDI pipeline for higher quality output.

    Render a layer by placing samples on the rhythm grid.

    Args:
        rhythm: RhythmPattern with quantized beats
        sample_mapping: Dict mapping instrument names to sample file paths
        bpm: BPM to use (default: from rhythm pattern)
        output_dir: Directory to save output (default: from settings)

    Returns:
        Path to the rendered WAV file
    """
    bpm = bpm or rhythm.bpm

    # Calculate timing
    seconds_per_beat = 60.0 / bpm
    seconds_per_subdivision = seconds_per_beat / (rhythm.subdivisions / 4)
    ms_per_subdivision = seconds_per_subdivision * 1000

    # Calculate total duration
    total_subdivisions = rhythm.bars * rhythm.subdivisions
    total_ms = int(total_subdivisions * ms_per_subdivision)

    # Get all required sample files from cache/R2
    sample_cache: dict[str, AudioSegment] = {}
    for beat in rhythm.beats:
        instrument_name = beat.instrument.value
        if (
            instrument_name in sample_mapping
            and sample_mapping[instrument_name] not in sample_cache
        ):
            try:
                # sample_mapping now contains local file paths for samples
                file_path = Path(sample_mapping[instrument_name])
                if file_path.exists():
                    sample_cache[sample_mapping[instrument_name]] = (
                        AudioSegment.from_file(str(file_path))
                    )
            except Exception:
                continue

    # Create silent base track
    output = AudioSegment.silent(duration=total_ms)

    for beat in rhythm.beats:
        instrument_name = beat.instrument.value

        if instrument_name not in sample_mapping:
            continue

        sample_path = sample_mapping[instrument_name]

        if sample_path not in sample_cache:
            continue

        sample = sample_cache[sample_path]

        # Apply velocity (volume adjustment)
        if beat.velocity < 1.0:
            # Convert velocity to dB adjustment (-20dB at velocity 0)
            db_adjustment = (beat.velocity - 1.0) * 20
            sample = sample + db_adjustment

        # Calculate position in milliseconds
        total_position = (beat.bar * rhythm.subdivisions) + beat.position
        position_ms = int(total_position * ms_per_subdivision)

        # Overlay the sample
        if position_ms < len(output):
            output = output.overlay(sample, position=position_ms)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output.export(tmp.name, format="wav")
        tmp_path = Path(tmp.name)

    try:
        # Upload to R2
        r2_key = f"layers/{uuid.uuid4().hex[:8]}.wav"
        url = await storage.put_file(tmp_path, r2_key, background=True)
        return url
    finally:
        tmp_path.unlink(missing_ok=True)


async def mix_project(
    project: Project,
    storage: FileStorageService,
    output_dir: Path | None = None,
) -> str:
    """
    Mix all layers in a project into a final MP3.

    Args:
        project: Project with layers to mix
        output_dir: Directory to save output (default: from settings)

    Returns:
        Path to the mixed MP3 file
    """
    if not project.layers:
        raise ValueError("Project has no layers to mix")

    # Load all layer audio files from cache/R2 with their bar offsets
    layers_with_offset: list[tuple[AudioSegment, int]] = []
    max_duration = 0

    for layer in project.layers:
        if layer.audio_file:
            try:
                # Get file from cache or download from R2
                if layer.audio_file.startswith("http"):
                    # Extract R2 key from URL (simplified)
                    r2_key = layer.audio_file.split("/")[-1]
                else:
                    r2_key = layer.audio_file

                file_path = await storage.get_file(r2_key)
                audio = AudioSegment.from_file(str(file_path))
                # Convert start_bar to milliseconds offset
                # 4 beats per bar (4/4 time), seconds_per_beat = 60/bpm
                offset_ms = int(layer.start_bar * 4 * (60.0 / project.bpm) * 1000)
                layers_with_offset.append((audio, offset_ms))
                max_duration = max(max_duration, offset_ms + len(audio))
            except Exception:
                pass

    if not layers_with_offset:
        raise ValueError("No valid layer audio files found")

    # Create base track with max duration (stereo 44100Hz to match rendered layers)
    mixed = AudioSegment.silent(duration=max_duration, frame_rate=44100).set_channels(2)

    # Overlay all layers at their respective positions
    for layer_audio, offset_ms in layers_with_offset:
        # Normalize format before overlay to avoid sample rate / channel mismatches
        if layer_audio.frame_rate != 44100:
            layer_audio = layer_audio.set_frame_rate(44100)
        if layer_audio.channels != 2:
            layer_audio = layer_audio.set_channels(2)
        mixed = mixed.overlay(layer_audio, position=offset_ms)

    # Normalize to -3 dBFS
    target_dbfs = -3.0
    change_in_dbfs = target_dbfs - mixed.dBFS
    mixed = mixed.apply_gain(change_in_dbfs)

    # Save to temp file
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project.name)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mixed.export(tmp.name, format="mp3")
        tmp_path = Path(tmp.name)

    try:
        # Upload to R2
        r2_key = f"projects/{safe_name}_{project.id[:8]}.mp3"
        url = await storage.put_file(tmp_path, r2_key, background=True)
        return url
    finally:
        tmp_path.unlink(missing_ok=True)


def create_empty_layer_audio(
    duration_bars: int,
    bpm: int,
    subdivisions: int = 16,
    output_dir: Path | None = None,
) -> str:
    """
    Create an empty (silent) layer audio file.

    Args:
        duration_bars: Number of bars
        bpm: Beats per minute
        subdivisions: Subdivisions per bar
        output_dir: Output directory

    Returns:
        Path to the silent WAV file
    """
    out_dir = Path(output_dir or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seconds_per_beat = 60.0 / bpm
    seconds_per_subdivision = seconds_per_beat / (subdivisions / 4)
    total_ms = int(duration_bars * subdivisions * seconds_per_subdivision * 1000)

    silent = AudioSegment.silent(duration=total_ms)

    filename = f"empty_{uuid.uuid4().hex[:8]}.wav"
    output_path = out_dir / filename

    silent.export(str(output_path), format="wav")

    return str(output_path)
