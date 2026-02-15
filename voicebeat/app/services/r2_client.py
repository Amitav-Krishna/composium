import asyncio
import sys
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings


class R2Client:
    """Simple R2 client using boto3 for Cloudflare R2 operations."""

    def __init__(self):
        if boto3 is None:
            raise ImportError(
                "boto3 is required for R2 operations. Install with: pip install boto3"
            )

        self.s3: Any = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
        )
        self.bucket: str = settings.r2_bucket_name

    async def upload_file(self, file_path: str, r2_key: str) -> str:
        """Upload file to R2 and return public URL"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.s3.upload_file, file_path, self.bucket, r2_key
        )
        return f"https://{self.bucket}.r2.dev/{r2_key}"

    async def download_file(self, r2_key: str, local_path: str) -> bool:
        """Download file from R2 to local path"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.s3.download_file, self.bucket, r2_key, local_path
            )
            return True
        except ClientError:
            return False

    def generate_presigned_url(self, r2_key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for direct access"""
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": r2_key},
            ExpiresIn=expires_in,
        )
