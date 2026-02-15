#!/usr/bin/env python3
"""
R2 Setup Script for User-Generated Content

This script helps set up Cloudflare R2 storage for VoiceBeat by:
1. Validating the R2 configuration
2. Testing upload/download functionality

NOTE: This script does NOT handle sample files - samples are always local.
R2 is only used for user-generated content like recordings, layers, and outputs.

Sample files should be managed using:
- scripts/generate_samples.py (to create test samples)
- scripts/build_catalog.py (to rebuild catalog from local files)

Usage:
    python scripts/setup_r2.py [--validate] [--test] [--all]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.r2_client import R2Client
from config.settings import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Sample upload functions removed - samples are now local-only
# R2 is only used for user-generated content (recordings, layers, outputs)


async def validate_r2_config():
    """Validate R2 configuration and connectivity."""
    logger.info("Validating R2 configuration...")

    # Check required settings
    required_settings = [
        "r2_account_id",
        "r2_access_key_id",
        "r2_secret_access_key",
        "r2_bucket_name",
    ]

    missing = []
    for setting in required_settings:
        value = getattr(settings, setting, "")
        if not value:
            missing.append(setting.upper())

    if missing:
        logger.error(f"Missing required R2 settings: {', '.join(missing)}")
        logger.error("Please add these to your .env file:")
        for setting in missing:
            logger.error(f"  {setting}=your_value_here")
        return False

    # Test connectivity by trying to list bucket
    try:
        r2_client = R2Client()
        # Try to generate a presigned URL as a connectivity test
        test_url = r2_client.generate_presigned_url("test-connectivity", expires_in=60)
        logger.info("✓ R2 connectivity test passed")
        logger.info(f"  Account ID: {settings.r2_account_id}")
        logger.info(f"  Bucket: {settings.r2_bucket_name}")
        return True

    except Exception as e:
        logger.error(f"R2 connectivity test failed: {e}")
        logger.error("Please check your R2 credentials and bucket configuration")
        return False


async def test_upload_download():
    """Test upload and download functionality."""
    logger.info("Testing upload/download functionality...")

    try:
        r2_client = R2Client()

        # Create a test file
        test_content = b"VoiceBeat R2 test file"
        test_file = Path("test_r2.txt")
        test_file.write_bytes(test_content)

        try:
            # Upload test file
            logger.info("Testing upload...")
            await r2_client.upload_file(str(test_file), "test/test_r2.txt")
            logger.info("✓ Upload test passed")

            # Download test file
            logger.info("Testing download...")
            download_path = Path("test_r2_download.txt")
            success = await r2_client.download_file(
                "test/test_r2.txt", str(download_path)
            )

            if success and download_path.exists():
                downloaded_content = download_path.read_bytes()
                if downloaded_content == test_content:
                    logger.info("✓ Download test passed")
                    return True
                else:
                    logger.error("Downloaded content doesn't match uploaded content")
            else:
                logger.error("Download test failed")

        finally:
            # Cleanup test files
            test_file.unlink(missing_ok=True)
            Path("test_r2_download.txt").unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Upload/download test failed: {e}")
        return False

    return False


async def main():
    parser = argparse.ArgumentParser(description="Set up R2 storage for VoiceBeat")
    # Sample upload options removed - samples are local-only
    parser.add_argument(
        "--validate", action="store_true", help="Validate R2 configuration"
    )
    parser.add_argument("--test", action="store_true", help="Run upload/download test")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")

    args = parser.parse_args()

    if not any([args.validate, args.test, args.all]):
        parser.print_help()
        return

    logger.info("=" * 60)
    logger.info("VoiceBeat R2 Setup Script")
    logger.info("=" * 60)

    success = True

    # Validation first
    if args.validate or args.all:
        if not await validate_r2_config():
            logger.error("R2 configuration validation failed")
            success = False
            return

    # Test upload/download
    if args.test or args.all:
        if not await test_upload_download():
            logger.error("R2 upload/download test failed")
            success = False

    # Sample uploads removed - samples are local-only
    if args.all:
        logger.info("Skipping sample upload - samples are managed locally")
        logger.info("Use scripts/build_catalog.py to rebuild sample catalog")

    if success:
        logger.info("=" * 60)
        logger.info("✓ R2 setup completed successfully!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Set USE_R2_STORAGE=true in your .env file")
        logger.info("2. Ensure sample files exist in local samples/ directory")
        logger.info("3. Run scripts/build_catalog.py to update sample catalog")
        logger.info("4. Start the VoiceBeat server with 'python app/main.py'")
    else:
        logger.error("=" * 60)
        logger.error("✗ R2 setup completed with errors")
        logger.error("Please check the logs above and resolve any issues")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
