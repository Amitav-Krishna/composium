#!/usr/bin/env python3
"""
Test script to verify local sample lookup functionality.

This test creates a minimal sample directory structure and verifies
that the SampleLookup class can find and return local file paths.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.schemas import Genre, Instrument, MusicDescription
from app.services.sample_lookup import SampleLookup, get_sample_catalog


def create_test_samples(temp_dir: Path):
    """Create a minimal test sample structure."""
    # Create sample directories and files
    genres = ["hip-hop", "electronic"]
    instruments = ["kick", "snare", "bass"]

    for genre in genres:
        for instrument in instruments:
            sample_dir = temp_dir / genre / instrument
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Create a dummy WAV file
            sample_file = sample_dir / f"{instrument}_01.wav"
            sample_file.write_bytes(b"dummy wav content")
            print(f"Created: {sample_file}")


def test_sample_lookup():
    """Test the SampleLookup class with local files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temp directory: {temp_path}")

        # Create test samples
        create_test_samples(temp_path)

        # Override settings for test
        import app.services.sample_lookup

        original_samples_dir = app.services.sample_lookup.settings.samples_dir
        app.services.sample_lookup.settings.samples_dir = temp_path

        try:
            # Initialize sample lookup
            lookup = SampleLookup()
            lookup.initialize()

            print(f"Sample catalog: {lookup.sample_catalog}")

            # Test getting individual samples
            kick_path = lookup.get_sample("hip-hop", "kick")
            print(f"Hip-hop kick: {kick_path}")
            assert kick_path is not None
            assert Path(kick_path).exists()

            # Test getting samples with fallback
            synth_path = lookup.get_sample(
                "jazz", "bass"
            )  # jazz doesn't exist, should fallback
            print(f"Jazz bass (fallback): {synth_path}")

            # Test lookup_samples method
            description = MusicDescription(
                genre=Genre.HIP_HOP,
                instruments=[Instrument.KICK, Instrument.SNARE, Instrument.BASS],
            )

            sample_mapping = lookup.lookup_samples(description)
            print(f"Sample mapping: {sample_mapping}")

            assert len(sample_mapping) == 3
            for instrument, path in sample_mapping.items():
                assert Path(path).exists(), f"Sample file not found: {path}"
                print(f"✓ {instrument}: {path}")

            print("✓ SampleLookup tests passed!")

        finally:
            # Restore original settings
            app.services.sample_lookup.settings.samples_dir = original_samples_dir


def test_catalog_function():
    """Test the get_sample_catalog function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temp directory for catalog test: {temp_path}")

        # Create test samples
        create_test_samples(temp_path)

        # Test catalog generation
        catalog = get_sample_catalog(temp_path)
        print(f"Generated catalog: {json.dumps(catalog, indent=2)}")

        assert "genres" in catalog
        assert "instruments" in catalog
        assert "samples" in catalog

        assert len(catalog["genres"]) == 2
        assert "hip-hop" in catalog["genres"]
        assert "electronic" in catalog["genres"]

        assert len(catalog["instruments"]) == 3
        assert "kick" in catalog["instruments"]
        assert "snare" in catalog["instruments"]
        assert "bass" in catalog["instruments"]

        # Check samples structure
        assert "hip-hop" in catalog["samples"]
        assert "kick" in catalog["samples"]["hip-hop"]
        assert len(catalog["samples"]["hip-hop"]["kick"]) == 1
        assert catalog["samples"]["hip-hop"]["kick"][0] == "kick_01.wav"

        print("✓ Catalog function tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Local Sample Lookup")
    print("=" * 60)

    try:
        test_sample_lookup()
        print()
        test_catalog_function()

        print("=" * 60)
        print("✓ All tests passed!")
        print("Local sample lookup is working correctly.")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
