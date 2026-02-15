import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .cache_manager import CacheManager
from .r2_client import R2Client


class FileStorageService:
    """File storage service combining R2 and local cache."""

    def __init__(self, cache_dir: str = "cache", cache_ttl_hours: int = 24):
        self.r2_client: R2Client = R2Client()
        self.cache: CacheManager = CacheManager(cache_dir, cache_ttl_hours)
        self.upload_queue: asyncio.Queue[tuple[Path, str]] = asyncio.Queue()
        self._upload_worker_task: Optional[Any] = None

    async def start_background_uploader(self):
        """Start background upload worker"""
        if self._upload_worker_task is None:
            self._upload_worker_task = asyncio.create_task(self._upload_worker())

    async def _upload_worker(self):
        """Background worker for uploads"""
        while True:
            try:
                file_path, r2_key = await self.upload_queue.get()
                await self.r2_client.upload_file(str(file_path), r2_key)
                print(f"Uploaded {r2_key} to R2")
            except Exception as e:
                print(f"Upload failed: {e}")
            finally:
                self.upload_queue.task_done()

    async def get_file(self, r2_key: str) -> Path:
        """Get file, from cache if available, otherwise download"""
        # Try cache first
        cached_path = self.cache.get(r2_key)
        if cached_path:
            return cached_path

        # Download to cache
        cache_path = self.cache._get_cache_path(r2_key)
        success = await self.r2_client.download_file(r2_key, str(cache_path))

        if success:
            self.cache.metadata[r2_key] = time.time()
            self.cache._save_metadata()
            return cache_path
        else:
            raise FileNotFoundError(f"Could not download {r2_key} from R2")

    async def put_file(
        self, file_path: Path, r2_key: str, background: bool = True
    ) -> str:
        """Store file, optionally upload in background"""
        # Add to cache immediately
        self.cache.put(r2_key, file_path)

        if background:
            # Queue for background upload
            await self.upload_queue.put((file_path, r2_key))
            # Return presigned URL immediately
            return self.r2_client.generate_presigned_url(r2_key)
        else:
            # Upload immediately
            return await self.r2_client.upload_file(str(file_path), r2_key)

    def cleanup_cache(self):
        """Clean up expired cache entries"""
        self.cache.cleanup_expired()
