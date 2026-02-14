import uuid
from pathlib import Path
from pydub import AudioSegment
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings
from app.models.schemas import RhythmPattern, Project


def render_layer(
    rhythm: RhythmPattern,
    sample_mapping: dict[str, str],
    bpm: int | None = None,
    output_dir: Path | None = None,
) -> str:
    """
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
    out_dir = Path(output_dir or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculate timing
    seconds_per_beat = 60.0 / bpm
    seconds_per_subdivision = seconds_per_beat / (rhythm.subdivisions / 4)
    ms_per_subdivision = seconds_per_subdivision * 1000

    # Calculate total duration
    total_subdivisions = rhythm.bars * rhythm.subdivisions
    total_ms = int(total_subdivisions * ms_per_subdivision)

    # Create silent base track
    output = AudioSegment.silent(duration=total_ms)

    # Load samples (cache to avoid reloading)
    sample_cache: dict[str, AudioSegment] = {}

    for beat in rhythm.beats:
        instrument_name = beat.instrument.value

        if instrument_name not in sample_mapping:
            continue

        sample_path = sample_mapping[instrument_name]

        # Load sample if not cached
        if sample_path not in sample_cache:
            try:
                sample_cache[sample_path] = AudioSegment.from_file(sample_path)
            except Exception:
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

    # Generate unique filename
    filename = f"layer_{uuid.uuid4().hex[:8]}.wav"
    output_path = out_dir / filename

    # Export as WAV
    output.export(str(output_path), format="wav")

    return str(output_path)


def mix_project(
    project: Project,
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
    out_dir = Path(output_dir or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not project.layers:
        raise ValueError("Project has no layers to mix")

    # Load all layer audio files
    layers: list[AudioSegment] = []
    max_duration = 0

    for layer in project.layers:
        if layer.audio_file:
            try:
                audio = AudioSegment.from_file(layer.audio_file)
                layers.append(audio)
                max_duration = max(max_duration, len(audio))
            except Exception:
                pass

    if not layers:
        raise ValueError("No valid layer audio files found")

    # Create base track with max duration
    mixed = AudioSegment.silent(duration=max_duration)

    # Overlay all layers
    for layer_audio in layers:
        mixed = mixed.overlay(layer_audio)

    # Normalize to -3 dBFS
    target_dbfs = -3.0
    change_in_dbfs = target_dbfs - mixed.dBFS
    mixed = mixed.apply_gain(change_in_dbfs)

    # Generate output filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project.name)
    filename = f"{safe_name}_{project.id[:8]}.mp3"
    output_path = out_dir / filename

    # Export as MP3
    mixed.export(str(output_path), format="mp3")

    return str(output_path)


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
