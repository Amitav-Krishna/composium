# R2 Integration for VoiceBeat

This document explains the Cloudflare R2 object storage integration for VoiceBeat, which replaces local file storage with a cloud-based solution for better scalability and reliability.

## Architecture

```
User Request → Local Cache Check → R2 Download (if miss) → Process → Background R2 Upload
```

### Core Components

1. **R2 Client** (`app/services/r2_client.py`) - Simple boto3 wrapper for R2 operations
2. **Cache Manager** (`app/services/cache_manager.py`) - Git-ignored local cache with TTL expiration
3. **File Storage Service** (`app/services/file_storage.py`) - Combines R2 and cache for optimal performance
4. **Background Upload Queue** - Async upload queue for non-blocking operations

## Setup

### 1. Install Dependencies

The R2 integration requires boto3:

```bash
pip install boto3==1.35.0
```

### 2. Environment Configuration

Add the following to your `.env` file:

```bash
# Cloudflare R2 Configuration
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=voicebeat-audio

# Cache Configuration (optional)
CACHE_DIR=cache
CACHE_TTL_HOURS=24

# Feature Flags (optional)
USE_R2_STORAGE=true
FALLBACK_TO_LOCAL=true
```

### 3. R2 Bucket Setup

Create a Cloudflare R2 bucket and obtain your credentials from the Cloudflare dashboard:

1. Go to R2 Object Storage in your Cloudflare dashboard
2. Create a new bucket (e.g., `voicebeat-audio`)
3. Create API tokens with read/write permissions
4. Configure the bucket for public access if needed

### 4. Upload Sample Catalog and Files

Use the setup script to upload your sample catalog and files:

```bash
# Validate R2 configuration
python scripts/setup_r2.py --validate

# Upload sample catalog and files
python scripts/setup_r2.py --all
```

## File Structure

```
voicebeat/
├── app/services/
│   ├── r2_client.py          # R2 operations (user content only)
│   ├── cache_manager.py      # Local file cache
│   ├── file_storage.py       # Combined storage service (user content)
│   └── sample_lookup.py      # Local sample file lookup
├── catalog/
│   └── samples.json          # Local sample catalog (no R2)
├── samples/                  # Local sample library (never R2)
├── cache/                    # Local cache (git-ignored)
└── scripts/
    ├── setup_r2.py           # R2 setup utility (user content)
    └── build_catalog.py      # Local sample catalog builder
```

## Usage

### File Storage Service

The `FileStorageService` is the main interface for all file operations:

```python
from app.services.file_storage import FileStorageService

# Initialize
storage = FileStorageService()
await storage.start_background_uploader()

# Store file (uploads to R2 in background)
url = await storage.put_file(local_path, "audio/segment.wav")

# Retrieve file (from cache or download from R2)
cached_path = await storage.get_file("audio/segment.wav")

# Clean up expired cache entries
storage.cleanup_cache()
```

### Audio Segment Storage

Audio segments are automatically stored in R2 with the following key structure:

- Segments: `segments/{segment_id}.wav`
- Layers: `layers/{layer_id}.wav`
- Projects: `projects/{project_name}_{project_id}.mp3`
- Rhythms: `rhythms/{uuid}.mp3`
- Melodies: `melodies/{uuid}.mp3`

### Sample Management

**Sample files are ALWAYS stored locally** - they never use R2 storage.

- Local samples: `samples/{genre}/{instrument}/{sample_name}.wav`
- Local catalog: `catalog/samples.json` (contains only `local_path`, no R2 keys)
- Use `scripts/build_catalog.py` to regenerate catalog from local files

R2 storage is only used for:
- User recordings (`segments/`)
- Generated layers (`layers/`)
- Mixed project outputs (`projects/`)
- Other user-generated content

## Cache Management

### Local Cache

- Location: `cache/` directory (git-ignored)
- TTL: 24 hours (configurable)
- Metadata: Stored in `cache/metadata.json`
- Cleanup: Automatic hourly cleanup + manual cleanup on startup

### Cache Behavior

1. **Cache Hit**: File served immediately from local cache
2. **Cache Miss**: File downloaded from R2 and cached locally
3. **Cache Expiry**: Files older than TTL are removed automatically
4. **Storage**: New files are cached immediately and uploaded to R2 in background

## Performance Benefits

### Cache-First Approach

- **Local Speed**: Frequently accessed files served at local disk speeds
- **Background Uploads**: New files uploaded asynchronously, no blocking
- **Presigned URLs**: Direct browser access for immediate playback

### Scalability

- **Unlimited Storage**: R2 handles storage scaling automatically
- **Global Distribution**: R2's CDN provides worldwide performance
- **Cost Efficient**: Only active files consume local cache space

## Monitoring

### Simple Metrics

Monitor these aspects of your R2 integration:

```python
# Check cache hit rate
cache_hits = len([f for f in cache.metadata if cache.get(f)])
total_requests = cache_hits + download_attempts

# Monitor upload queue
queue_size = storage.upload_queue.qsize()

# Check cache size
cache_size = sum(f.stat().st_size for f in cache.cache_dir.glob("*.cache"))
```

### Alerts

Set up alerts for:

- Cache directory > 5GB
- Upload queue > 100 items  
- High cache miss rate (>50%)
- Failed upload count increasing

## Troubleshooting

### Common Issues

**1. R2 Connection Failed**
```bash
# Check credentials
python scripts/setup_r2.py --validate

# Test connectivity
python scripts/setup_r2.py --test
```

**2. Cache Directory Full**
```bash
# Manual cleanup
rm -rf cache/

# Or in Python
storage.cleanup_cache()
```

**3. Upload Queue Backing Up**
```bash
# Check upload worker is running
# Look for "Uploaded {key} to R2" in logs
# Restart background uploader if needed
```

### Development Mode

For development without R2:

```bash
USE_R2_STORAGE=false
FALLBACK_TO_LOCAL=true
```

This falls back to local storage in the `output/` directory.

## Migration from Local Storage

### Existing Projects

Existing local files will continue to work. The system will:

1. Check cache first (empty initially)
2. Fall back to local files if R2 fails
3. Gradually migrate to R2 as new files are created

### Sample Files

Upload your existing samples to R2:

```bash
python scripts/setup_r2.py --upload-samples
```

## Security Considerations

### Access Control

- Use dedicated R2 API tokens with minimal permissions
- Rotate API keys regularly
- Monitor R2 access logs for unusual activity

### Data Protection

- All uploads use HTTPS encryption in transit
- R2 provides encryption at rest
- Presigned URLs expire automatically (default 1 hour)

### Local Cache Security

- Cache directory should not be web-accessible
- Consider disk encryption for sensitive audio content
- Regular cache cleanup prevents data accumulation

## Cost Optimization

### Storage Costs

- R2 charges per GB stored and operations
- Cache reduces repeated downloads
- Background uploads batch operations efficiently

### Best Practices

- Set reasonable cache TTL (24 hours default)
- Monitor cache hit rates
- Clean up old projects periodically
- Use presigned URLs for direct browser access

## API Integration

### FastAPI Integration

The storage service integrates seamlessly with FastAPI:

```python
# In routes
storage = request.app.state.storage

# For file uploads
url = await storage.put_file(temp_file, r2_key, background=True)

# For file downloads  
cached_file = await storage.get_file(r2_key)
return FileResponse(cached_file)
```

### Error Handling

```python
try:
    file_path = await storage.get_file(r2_key)
except FileNotFoundError:
    # Handle missing file
    raise HTTPException(status_code=404, detail="Audio file not found")
except Exception as e:
    # Handle R2 errors
    logger.error(f"R2 error: {e}")
    raise HTTPException(status_code=500, detail="Storage error")
```

This integration provides a robust, scalable storage solution while maintaining the performance characteristics needed for real-time audio processing.