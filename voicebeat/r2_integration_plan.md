# VoiceBeat R2 Integration Plan

## Overview

Replace local file storage with Cloudflare R2 object storage across VoiceBeat's entire audio pipeline. This addresses scalability, reliability, and deployment issues with the current local filesystem approach.

## Current File Storage Analysis

### File Types & Locations
1. **Audio Segments** - Extracted from user recordings (`extract_segment_audio()`)
2. **Generated Audio Layers** - Rendered rhythm/melody tracks 
3. **Mixed Project Audio** - Final combined MP3 outputs
4. **TTS Audio Files** - Generated speech feedback
5. **Sample Library** - Instrument samples for composition
6. **User Uploaded Files** - Original recordings
7. **MIDI Files** - Generated notation files

### Current File Operations Inventory

#### Read Operations
- `app/services/sample_lookup.py` - Reading sample files from `samples/` directory
- `app/services/track_assembler.py` - Loading audio files with `AudioSegment.from_file()`
- `app/services/subagents/base.py` - Audio duration calculation
- `app/api/routes.py` - File download endpoint (`/download/{filename}`)

#### Write Operations
- `app/services/segmenter.py` - Audio segment extraction
- `app/services/track_assembler.py` - Layer rendering and project mixing
- `app/services/tts.py` - TTS output files
- `app/services/vocal_processor.py` - Vocal processing outputs
- `app/services/composium_bridge.py` - Melody/rhythm rendering
- `app/services/notation.py` - MIDI file generation

## Implementation Plan

### Phase 1: Core R2 Service Layer

#### Task 1.1: Create R2 Client Service
**File**: `app/services/r2_client.py`
- Initialize boto3 S3 client with R2 credentials
- Implement upload/download/delete operations
- Add presigned URL generation
- Handle connection pooling and retries

```python
class R2Client:
    async def upload_file(self, file_data: bytes, key: str, content_type: str) -> str
    async def download_file(self, key: str) -> bytes
    async def delete_file(self, key: str) -> bool
    async def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str
    async def file_exists(self, key: str) -> bool
```

#### Task 1.2: Update Configuration
**File**: `config/settings.py`
- Validate R2 configuration on startup
- Add bucket name, region settings
- Add presigned URL expiration settings

#### Task 1.3: Create Storage Abstraction Layer
**File**: `app/services/storage.py`
- Abstract interface for file operations
- Fallback to local storage for development
- Consistent error handling

### Phase 2: Audio Pipeline Integration

#### Task 2.1: Update Audio Segment Extraction
**File**: `app/services/segmenter.py`
**Function**: `extract_segment_audio()`

**Current**:
```python
output_path = out_dir / filename
sf.write(str(output_path), chunk, sr)
return str(output_path)
```

**New**:
```python
# Write to buffer instead of file
buffer = io.BytesIO()
sf.write(buffer, chunk, sr, format='WAV')
buffer.seek(0)

# Upload to R2
r2_key = f"segments/{segment.id}.wav"
await r2_client.upload_file(buffer.getvalue(), r2_key, "audio/wav")

# Return R2 URL
segment.audio_file = await r2_client.generate_presigned_url(r2_key)
return segment.audio_file
```

#### Task 2.2: Update Track Assembler
**File**: `app/services/track_assembler.py`

**Functions to update**:
- `render_layer()` - Upload rendered layers to R2
- `mix_project()` - Upload mixed audio to R2
- `create_empty_layer_audio()` - Upload silent tracks to R2

**Key changes**:
- Replace `AudioSegment.from_file(local_path)` with R2 downloads
- Replace `.export(local_path)` with R2 uploads
- Update return values to R2 URLs

#### Task 2.3: Update Composium Bridge
**File**: `app/services/composium_bridge.py`

**Functions**: `render_melody()`, `render_rhythm()`
- Upload rendered audio to R2 instead of local filesystem
- Return R2 presigned URLs

#### Task 2.4: Update Vocal Processor
**File**: `app/services/vocal_processor.py`

**Functions**: `autotune()`, `pitch_shift_words()`, `beat_snap_words()`, `tts_melodic_vocal()`, `tts_rhythmic_vocal()`
- Replace local file exports with R2 uploads
- Return R2 URLs for generated vocals

#### Task 2.5: Update TTS Service
**File**: `app/services/tts.py`
**Function**: `speak_to_file()`

**Current**:
```python
output = Path(output_path)
output.write_bytes(audio_bytes)
return output
```

**New**:
```python
r2_key = f"tts/{uuid.uuid4().hex}.wav"
await r2_client.upload_file(audio_bytes, r2_key, "audio/wav")
return await r2_client.generate_presigned_url(r2_key)
```

### Phase 3: Sample Library Migration

#### Task 3.1: Migrate Sample Files to R2
**Script**: `scripts/migrate_samples_to_r2.py`
- Upload all files from `samples/` directory to R2
- Maintain directory structure as R2 key prefixes
- Generate catalog with R2 URLs

#### Task 3.2: Update Sample Lookup
**File**: `app/services/sample_lookup.py`

**Current**:
```python
base_dir = samples_dir or settings.samples_dir
# File system traversal
```

**New**:
```python
# Fetch sample catalog from R2 or cache
sample_urls = await get_r2_sample_catalog()
```

### Phase 4: API Endpoint Updates

#### Task 4.1: Update File Download Endpoint
**File**: `app/api/routes.py`
**Endpoint**: `GET /download/{filename}`

**Replace**: Local `FileResponse` 
**With**: Redirect to R2 presigned URL or proxy download

#### Task 4.2: Update File Upload Handling
**File**: `app/api/routes.py`

**Endpoints affected**:
- `POST /api/v1/process`
- `POST /api/v1/describe`  
- `POST /api/v1/rhythm`
- `POST /api/v1/projects/{id}/layers`

**Changes**:
- Upload received files directly to R2
- Pass R2 URLs through processing pipeline
- Update response models with R2 URLs

#### Task 4.3: Update Project Mixing
**File**: `app/api/routes.py`
**Endpoint**: `GET /api/v1/projects/{id}/mix`

**Current**: Returns `FileResponse` with local file
**New**: Return R2 presigned URL or stream from R2

### Phase 5: Data Model Updates

#### Task 5.1: Update Schema Models
**File**: `app/models/schemas.py`

**Models to update**:
```python
class AudioSegment(BaseModel):
    audio_file: Optional[str] = None  # Now R2 URL instead of local path

class Layer(BaseModel):
    audio_file: Optional[str] = None  # Now R2 URL

class Project(BaseModel):
    mixed_file: Optional[str] = None  # Now R2 URL
```

#### Task 5.2: URL Management
- Add helper functions to distinguish R2 URLs from local paths
- Handle URL expiration and regeneration
- Add metadata for file sizes and content types

### Phase 6: Error Handling & Resilience

#### Task 6.1: Implement Fallback Strategy
- Local cache for frequently accessed files
- Graceful degradation when R2 is unavailable
- Retry logic with exponential backoff

#### Task 6.2: Add Monitoring & Logging
- Track upload/download success rates
- Monitor R2 costs and usage
- Log file operation performance metrics

#### Task 6.3: File Lifecycle Management
- Implement automatic cleanup of old files
- Set TTL for temporary files
- Archive completed projects

## File-by-File Implementation Checklist

### Core Services
- [ ] `app/services/r2_client.py` - New R2 client service
- [ ] `app/services/storage.py` - New storage abstraction
- [ ] `config/settings.py` - Add R2 configuration validation

### Audio Processing Services
- [ ] `app/services/segmenter.py` - `extract_segment_audio()`
- [ ] `app/services/track_assembler.py` - `render_layer()`, `mix_project()`, `create_empty_layer_audio()`
- [ ] `app/services/composium_bridge.py` - `render_melody()`, `render_rhythm()`
- [ ] `app/services/vocal_processor.py` - All processing functions
- [ ] `app/services/tts.py` - `speak_to_file()`
- [ ] `app/services/notation.py` - `melody_to_midi()`

### Sample Management
- [ ] `app/services/sample_lookup.py` - `lookup_samples()`, `get_sample_catalog()`
- [ ] `scripts/migrate_samples_to_r2.py` - New migration script

### API Layer
- [ ] `app/api/routes.py` - All file upload/download endpoints
- [ ] `app/models/schemas.py` - Update URL fields

### Infrastructure
- [ ] `app/main.py` - Remove local directory creation
- [ ] `scripts/cleanup_r2_files.py` - New cleanup script

## Testing Strategy

### Unit Tests
- R2 client operations (mock S3 responses)
- Storage abstraction layer
- URL generation and validation

### Integration Tests  
- End-to-end audio processing pipeline
- File upload → processing → download flow
- Sample library access

### Performance Tests
- Large file upload/download performance
- Concurrent file operations
- Memory usage during file streaming

## Deployment Considerations

### Environment Variables
```
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=voicebeat-audio
R2_PUBLIC_URL=https://your-domain.r2.cloudflarestorage.com
```

### Migration Steps
1. Deploy R2 integration alongside existing file system
2. Run sample migration script
3. Update configuration to use R2 by default
4. Monitor for 24-48 hours
5. Remove local file system fallback

### Rollback Plan
- Keep local file system code as feature flag
- Ability to quickly switch back via configuration
- Backup of critical samples and user data

## Success Metrics

- **Reliability**: 99.9% file operation success rate
- **Performance**: File operations complete within 2x of local filesystem time
- **Cost**: R2 storage costs < $50/month for MVP usage
- **Scalability**: Support 1000+ concurrent users without filesystem limits

## Risks & Mitigations

### Risk: R2 Service Outage
**Mitigation**: Local cache + degraded mode operation

### Risk: Network Latency
**Mitigation**: Async operations + streaming for large files

### Risk: Cost Overruns
**Mitigation**: File lifecycle policies + usage monitoring

### Risk: Data Loss
**Mitigation**: Versioning enabled + backup strategy