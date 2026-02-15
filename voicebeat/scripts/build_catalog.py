#!/usr/bin/env python3
"""
Build Sample Catalog Script

This script scans the local samples directory and generates a catalog.json file
that maps genres and instruments to their available sample files.

Usage:
    python scripts/build_catalog.py [--output catalog/samples.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


def build_catalog(samples_dir: Path, output_file: Path):
    """Build catalog from local samples directory."""
    print(f"Scanning samples directory: {samples_dir}")

    if not samples_dir.exists():
        print(f"ERROR: Samples directory not found: {samples_dir}")
        return False

    catalog = {}
    total_samples = 0

    for genre_dir in sorted(samples_dir.iterdir()):
        if genre_dir.is_dir() and not genre_dir.name.startswith("."):
            genre_name = genre_dir.name
            print(f"  Processing genre: {genre_name}")
            catalog[genre_name] = {}

            for inst_dir in sorted(genre_dir.iterdir()):
                if inst_dir.is_dir() and not inst_dir.name.startswith("."):
                    inst_name = inst_dir.name
                    samples = []

                    for sample_file in sorted(inst_dir.glob("*.wav")):
                        samples.append(
                            {
                                "name": sample_file.name,
                                "local_path": str(sample_file.relative_to(Path("."))),
                            }
                        )
                        total_samples += 1

                    if samples:
                        catalog[genre_name][inst_name] = samples
                        print(f"    {inst_name}: {len(samples)} samples")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write catalog
    with open(output_file, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\nCatalog generated successfully!")
    print(f"  Total genres: {len(catalog)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Output file: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build sample catalog from local files"
    )
    parser.add_argument(
        "--output",
        default="catalog/samples.json",
        help="Output catalog file (default: catalog/samples.json)",
    )
    parser.add_argument(
        "--samples-dir", help="Samples directory (default: from settings)"
    )

    args = parser.parse_args()

    samples_dir = (
        Path(args.samples_dir) if args.samples_dir else Path(settings.samples_dir)
    )
    output_file = Path(args.output)

    print("=" * 60)
    print("VoiceBeat Sample Catalog Builder")
    print("=" * 60)

    success = build_catalog(samples_dir, output_file)

    if success:
        print("=" * 60)
        print("✓ Catalog built successfully!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ Failed to build catalog")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
