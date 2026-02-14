import pytest
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import RhythmPattern, QuantizedBeat, Instrument, Project, Layer, MusicDescription, Genre
from app.services.track_assembler import render_layer, mix_project, create_empty_layer_audio


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dir():
    """Get the samples directory."""
    return Path(__file__).parent.parent / "samples"


class TestRenderLayer:
    """Tests for rendering a layer."""

    def test_render_simple_pattern(self, temp_output_dir, sample_dir):
        """Test rendering a simple rhythm pattern."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=0, bar=0, instrument=Instrument.KICK, velocity=1.0),
                QuantizedBeat(position=4, bar=0, instrument=Instrument.SNARE, velocity=1.0),
                QuantizedBeat(position=8, bar=0, instrument=Instrument.KICK, velocity=1.0),
                QuantizedBeat(position=12, bar=0, instrument=Instrument.SNARE, velocity=1.0),
            ],
            bpm=120,
            bars=1,
        )

        sample_mapping = {
            "kick": str(sample_dir / "pop" / "kick" / "kick_01.wav"),
            "snare": str(sample_dir / "pop" / "snare" / "snare_01.wav"),
        }

        output_path = render_layer(pattern, sample_mapping, output_dir=temp_output_dir)

        assert Path(output_path).exists()
        assert output_path.endswith(".wav")

    def test_render_with_velocity(self, temp_output_dir, sample_dir):
        """Test rendering with different velocities."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=0, bar=0, instrument=Instrument.KICK, velocity=1.0),
                QuantizedBeat(position=4, bar=0, instrument=Instrument.KICK, velocity=0.5),
            ],
            bpm=120,
            bars=1,
        )

        sample_mapping = {
            "kick": str(sample_dir / "pop" / "kick" / "kick_01.wav"),
        }

        output_path = render_layer(pattern, sample_mapping, output_dir=temp_output_dir)

        assert Path(output_path).exists()

    def test_render_multi_bar_pattern(self, temp_output_dir, sample_dir):
        """Test rendering a multi-bar pattern."""
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=0, bar=0, instrument=Instrument.KICK, velocity=1.0),
                QuantizedBeat(position=0, bar=1, instrument=Instrument.KICK, velocity=1.0),
            ],
            bpm=120,
            bars=2,
        )

        sample_mapping = {
            "kick": str(sample_dir / "pop" / "kick" / "kick_01.wav"),
        }

        output_path = render_layer(pattern, sample_mapping, output_dir=temp_output_dir)

        assert Path(output_path).exists()

    def test_render_empty_pattern(self, temp_output_dir, sample_dir):
        """Test rendering an empty pattern creates silent audio."""
        pattern = RhythmPattern(
            beats=[],
            bpm=120,
            bars=1,
        )

        sample_mapping = {}

        output_path = render_layer(pattern, sample_mapping, output_dir=temp_output_dir)

        assert Path(output_path).exists()


class TestCreateEmptyLayerAudio:
    """Tests for creating empty layer audio."""

    def test_create_empty_audio(self, temp_output_dir):
        """Test creating empty audio file."""
        output_path = create_empty_layer_audio(
            duration_bars=2,
            bpm=120,
            output_dir=temp_output_dir,
        )

        assert Path(output_path).exists()
        assert output_path.endswith(".wav")


class TestMixProject:
    """Tests for mixing a project."""

    def test_mix_single_layer(self, temp_output_dir, sample_dir):
        """Test mixing a project with a single layer."""
        # First render a layer
        pattern = RhythmPattern(
            beats=[
                QuantizedBeat(position=0, bar=0, instrument=Instrument.KICK, velocity=1.0),
            ],
            bpm=120,
            bars=1,
        )

        sample_mapping = {
            "kick": str(sample_dir / "pop" / "kick" / "kick_01.wav"),
        }

        layer_path = render_layer(pattern, sample_mapping, output_dir=temp_output_dir)

        # Create project with layer
        layer = Layer(
            id="test-layer-1",
            description=MusicDescription(
                genre=Genre.POP,
                instruments=[Instrument.KICK],
            ),
            rhythm=pattern,
            sample_mapping=sample_mapping,
            audio_file=layer_path,
        )

        project = Project(
            id="test-project-1",
            name="Test Project",
            layers=[layer],
            bpm=120,
        )

        # Mix
        output_path = mix_project(project, output_dir=temp_output_dir)

        assert Path(output_path).exists()
        assert output_path.endswith(".mp3")

    def test_mix_multiple_layers(self, temp_output_dir, sample_dir):
        """Test mixing a project with multiple layers."""
        sample_mapping = {
            "kick": str(sample_dir / "pop" / "kick" / "kick_01.wav"),
            "snare": str(sample_dir / "pop" / "snare" / "snare_01.wav"),
        }

        # Create two layers
        layers = []
        for i, (pos, inst) in enumerate([(0, Instrument.KICK), (4, Instrument.SNARE)]):
            pattern = RhythmPattern(
                beats=[QuantizedBeat(position=pos, bar=0, instrument=inst, velocity=1.0)],
                bpm=120,
                bars=1,
            )
            layer_path = render_layer(pattern, sample_mapping, output_dir=temp_output_dir)

            layers.append(Layer(
                id=f"test-layer-{i}",
                description=MusicDescription(
                    genre=Genre.POP,
                    instruments=[inst],
                ),
                rhythm=pattern,
                sample_mapping=sample_mapping,
                audio_file=layer_path,
            ))

        project = Project(
            id="test-project-2",
            name="Multi Layer Test",
            layers=layers,
            bpm=120,
        )

        output_path = mix_project(project, output_dir=temp_output_dir)

        assert Path(output_path).exists()

    def test_mix_empty_project_raises_error(self, temp_output_dir):
        """Test that mixing an empty project raises an error."""
        project = Project(
            id="empty-project",
            name="Empty",
            layers=[],
            bpm=120,
        )

        with pytest.raises(ValueError, match="no layers"):
            mix_project(project, output_dir=temp_output_dir)
