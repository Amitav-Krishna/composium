# Sample Storage Changes

## Summary

Modified VoiceBeat to use **local-only storage for sample files** instead of R2 cloud storage. R2 is now only used for user-generated content (recordings, layers, mixed outputs).

## Key Changes

### 1. Sample Lookup Service (`app/services/sample_lookup.py`)

**Before:**
- Used `FileStorageService` to download samples from R2
- Async `initialize()` method to fetch catalog from R2
- Returned R2 keys that required cloud download

**After:**
- Removed R2 dependency completely
- Sync `initialize()` method builds catalog from local files
- Returns direct local file paths
- Falls back to scanning local directory if catalog missing

### 2. Sample Catalog (`catalog/samples.json`)

**Before:**
```json
{
  "electronic": {
    "bass": [{
      "name": "bass_01.wav",
      "r2_key": "samples/electronic/bass/bass_01.wav",
      "local_path": "samples/electronic/bass/bass_01.wav"
    }]
  }
}
```

**After:**
```json
{
  "electronic": {
    "bass": [{
      "name": "bass_01.wav",
      "local_path": "samples/electronic/bass/bass_01.wav"
    }]
  }
}
```

### 3. Track Assembler (`app/services/track_assembler.py`)

**Before:**
- Treated sample paths as R2 keys
- Used `storage.get_file()` to download from R2

**After:**
- Treats sample paths as local file paths
- Direct file system access with `Path.exists()` check

### 4. Application Startup (`app/main.py`)

**Before:**
```python
await app.state.samples.initialize()
```

**After:**
```python
app.state.samples.initialize()  # Now synchronous
```

### 5. Configuration (`config/settings.py`)

**Added clarification:**
```python
# Sample files are ALWAYS local - R2 is only used for user-generated content
# This includes: project audio, segment recordings, mixed outputs
# Sample library files are accessed directly from the local samples_dir
```

## New Scripts

### `scripts/build_catalog.py`
- Scans local `samples/` directory
- Generates `catalog/samples.json` with only local paths
- Replaces manual catalog maintenance

### Updated `scripts/setup_r2.py`
- Removed sample upload functionality
- Only handles user-generated content
- Added warnings about samples being local-only

## Storage Architecture

### Local Storage (Always Used)
```
samples/
├── hip-hop/
│   ├── kick/
│   │   └── kick_01.wav
│   └── snare/
│       └── snare_01.wav
└── electronic/
    ├── kick/
    │   └── kick_01.wav
    └── bass/
        └── bass_01.wav

catalog/
└── samples.json  # Generated from local files
```

### R2 Cloud Storage (User Content Only)
```
segments/          # Voice recordings
├── {segment_id}.wav
layers/            # Generated instrument tracks  
├── {layer_id}.wav
projects/          # Mixed outputs
└── {project_id}.mp3
```

## Benefits

1. **Faster Sample Access**: No network latency for sample loading
2. **Reduced R2 Costs**: Less cloud storage and bandwidth usage
3. **Simpler Development**: No R2 setup required for basic functionality
4. **Better Reliability**: Samples always available offline
5. **Easier Content Management**: Direct file system operations

## Migration Steps

1. **Existing Installations:**
   - Keep existing `samples/` directory
   - Run `scripts/build_catalog.py` to update catalog
   - Sample files in R2 can be removed (not used anymore)

2. **New Installations:**
   - Run `scripts/generate_samples.py` to create test samples
   - Run `scripts/build_catalog.py` to generate catalog
   - R2 setup only needed for user content features

## Backwards Compatibility

- Legacy `lookup_samples()` function still works
- `get_sample_catalog()` function unchanged
- API endpoints unchanged
- File paths in responses now point to local files

## Testing

Created `test_sample_lookup.py` to verify:
- Local file discovery works
- Catalog generation works
- Sample lookup returns valid paths
- Fallback mechanisms work

## Future Considerations

- Could add sample hot-reloading for development
- Could implement sample caching for frequently used files
- Could add sample validation (format, quality checks)
- Could support multiple sample library directories