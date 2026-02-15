import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CacheManager:
    """Simple file cache with TTL expiration."""

    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load cache metadata with timestamps"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def _get_cache_path(self, r2_key: str) -> Path:
        """Get local cache path for R2 key"""
        # Use hash to avoid filesystem issues with long/special paths
        key_hash = hashlib.md5(r2_key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, r2_key: str) -> Optional[Path]:
        """Get file from cache if exists and not expired"""
        cache_path = self._get_cache_path(r2_key)

        if not cache_path.exists():
            return None

        # Check TTL
        cached_time = self.metadata.get(r2_key, 0)
        if time.time() - cached_time > self.ttl_seconds:
            # Expired, remove from cache
            cache_path.unlink(missing_ok=True)
            self.metadata.pop(r2_key, None)
            self._save_metadata()
            return None

        return cache_path

    def put(self, r2_key: str, file_path: Path):
        """Put file in cache"""
        cache_path = self._get_cache_path(r2_key)

        # Copy file to cache
        shutil.copy2(file_path, cache_path)

        # Update metadata
        self.metadata[r2_key] = time.time()
        self._save_metadata()

    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for r2_key, cached_time in self.metadata.items():
            if current_time - cached_time > self.ttl_seconds:
                cache_path = self._get_cache_path(r2_key)
                cache_path.unlink(missing_ok=True)
                expired_keys.append(r2_key)

        for key in expired_keys:
            self.metadata.pop(key)

        if expired_keys:
            self._save_metadata()
