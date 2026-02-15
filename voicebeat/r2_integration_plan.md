# VoiceBeat R2 Integration Plan (Simplified)

## Overview

Replace local file storage with Cloudflare R2 object storage across VoiceBeat's entire audio pipeline, using a simple cache-first approach to maintain performance.

## Architecture

```
User Request → Local Cache Check → R2 Download (if miss) → Process → Background R2 Upload
```

### Core Components
1. **R2 Client** - Simple boto3 wrapper for R2 operations
2. **File Cache** - Git-ignored local cache with TTL expiration
3. **Background Uploader** - Async upload queue for non-blocking operations

## Implementation Plan

### Phase 1: Core Services

#### Task 1.1: Simple R2 Client
**File**: `app/services/r2_client.py`

```python
import boto3
from botocore.exceptions import ClientError
import asyncio
from typing import Optional
import aiofiles

class R2Client:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=f'https://{settings.r2_account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name='auto'
        )
        self.bucket = settings.r2_bucket_name
    
    async def upload_file(self, file_path: str, r2_key: str) -> str:
        """Upload file to R2 and return public URL"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            self.s3.upload_file, 
            file_path, self.bucket, r2_key
        )
        return f"https://{self.bucket}.r2.dev/{r2_key}"
    
    async def download_file(self, r2_key: str, local_path: str) -> bool:
        """Download file from R2 to local path"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3.download_file,
                self.bucket, r2_key, local_path
            )
            return True
        except ClientError:
            return False
    
    def generate_presigned_url(self, r2_key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for direct access"""
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': r2_key},
            ExpiresIn=expires_in
        )
```

#### Task 1.2: Simple Cache Manager
**File**: `app/services/cache_manager.py`

```python
import os
import time
from pathlib import Path
from typing import Optional
import hashlib
import json

class CacheManager:
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata with timestamps"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
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
        import shutil
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
```

#### Task 1.3: File Storage Service
**File**: `app/services/file_storage.py`

```python
import asyncio
from pathlib import Path
from typing import Optional
import tempfile
import uuid

class FileStorageService:
    def __init__(self):
        self.r2_client = R2Client()
        self.cache = CacheManager()
        self.upload_queue = asyncio.Queue()
        
    async def start_background_uploader(self):
        """Start background upload worker"""
        asyncio.create_task(self._upload_worker())
    
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
    
    async def put_file(self, file_path: Path, r2_key: str, background: bool = True) -> str:
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
```

### Phase 2: Update Audio Services

#### Task 2.1: Update Segmenter
**File**: `app/services/segmenter.py`

```python
async def extract_segment_audio(self, audio_data: np.ndarray, segment: AudioSegment) -> str:
    """Extract segment with R2 storage"""
    
    # Create temporary file for processing
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, chunk, sr)
        tmp_path = Path(tmp.name)
    
    # Store in R2 (background upload)
    r2_key = f"segments/{segment.id}.wav"
    url = await self.storage.put_file(tmp_path, r2_key, background=True)
    
    # Cleanup temp file
    tmp_path.unlink()
    
    return url
```

#### Task 2.2: Update Track Assembler
**File**: `app/services/track_assembler.py`

```python
async def render_layer(self, layer_data: LayerData) -> str:
    """Render layer with cached file access"""
    
    # Get all required files from cache/R2
    audio_files = []
    for segment in layer_data.segments:
        file_path = await self.storage.get_file(segment.audio_key)
        audio_files.append(AudioSegment.from_file(str(file_path)))
    
    # Process with local files (fast)
    rendered = self._combine_segments(audio_files)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        rendered.export(tmp.name, format="wav")
        tmp_path = Path(tmp.name)
    
    # Upload to R2
    r2_key = f"layers/{layer_data.id}.wav"
    url = await self.storage.put_file(tmp_path, r2_key, background=True)
    
    tmp_path.unlink()
    return url
```

#### Task 2.3: Update Sample Lookup
**File**: `app/services/sample_lookup.py`

```python
class SampleLookup:
    def __init__(self):
        self.storage = FileStorageService()
        # Keep sample catalog in memory for fast lookup
        self.sample_catalog = {}
    
    async def initialize(self):
        """Load sample catalog from R2"""
        # Download sample catalog (small JSON file)
        catalog_path = await self.storage.get_file("catalog/samples.json")
        with open(catalog_path) as f:
            self.sample_catalog = json.load(f)
    
    async def get_sample(self, sample_id: str) -> AudioSegment:
        """Get sample audio file"""
        sample_info = self.sample_catalog[sample_id]
        r2_key = sample_info['r2_key']
        
        # Get from cache or download
        file_path = await self.storage.get_file(r2_key)
        return AudioSegment.from_file(str(file_path))
```

### Phase 3: Configuration & Deployment

#### Task 3.1: Update Settings
**File**: `config/settings.py`

```python
class Settings:
    # R2 Configuration
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str = "voicebeat-audio"
    
    # Cache Configuration
    cache_dir: str = "cache"  # Git ignored
    cache_ttl_hours: int = 24
    
    # Feature Flags
    use_r2_storage: bool = True
    fallback_to_local: bool = True  # For development
```

#### Task 3.2: Update .gitignore
```
# Cache directory
cache/
*.cache

# Temporary files
temp/
tmp/
```

#### Task 3.3: Startup Integration
**File**: `app/main.py`

```python
@app.on_event("startup")
async def startup_event():
    # Initialize file storage
    app.state.storage = FileStorageService()
    await app.state.storage.start_background_uploader()
    
    # Initialize sample lookup
    app.state.samples = SampleLookup()
    await app.state.samples.initialize()
    
    # Start cache cleanup task
    asyncio.create_task(periodic_cache_cleanup())

async def periodic_cache_cleanup():
    """Clean cache every hour"""
    while True:
        await asyncio.sleep(3600)  # 1 hour
        app.state.storage.cleanup_cache()
```


## Deployment Steps

### 1. Environment Setup
```bash
# Environment variables
export R2_ACCOUNT_ID="your_account_id"
export R2_ACCESS_KEY_ID="your_access_key"
export R2_SECRET_ACCESS_KEY="your_secret_key"
export R2_BUCKET_NAME="voicebeat-audio"
```

```

### 3. Cache Management
- Cache directory is git-ignored
- Automatic cleanup every hour
- TTL of 24 hours for cache entries
- Manual cleanup: `rm -rf cache/`

## Benefits of This Approach

### Simplicity
- Single R2 client using standard boto3
- Simple file-based cache with JSON metadata
- Minimal configuration required

### Performance
- Cache-first approach maintains local file speeds
- Background uploads don't block user requests
- Presigned URLs for direct browser access

### Reliability
- Fallback to local storage during development
- Simple cache invalidation via TTL
- Easy rollback by disabling feature flag

### Cost Efficiency
- Only frequently accessed files stay in cache
- Automatic cleanup prevents disk bloat
- Background uploads reduce bandwidth costs

## Monitoring

### Simple Metrics
- Cache hit rate (log on miss)
- Upload queue length
- Cache directory size
- Failed upload count

### Basic Alerts
- Cache directory > 5GB
- Upload queue > 100 items
- High cache miss rate

This simplified approach provides R2 integration with performance benefits while keeping the implementation straightforward and maintainable.
