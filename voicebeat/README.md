# VoiceBeat

Voice-driven music creation tool. Record a SINGLE audio clip where you both **talk** (give instructions) and **hum/sing/beatbox** (the musical idea). VoiceBeat separates speech from music, interprets your instructions, and assembles a real track.

## Key Innovation

Unlike traditional approaches that require separate recordings for description and rhythm, VoiceBeat uses **timestamp-based audio segmentation**:

1. **Single Recording Input** - Talk and hum in one take
2. **Smart Segmentation** - Uses STT word timestamps to identify speech regions; gaps = music
3. **AI Orchestration** - OpenAI interprets instructions and assigns instruments using tool calls
4. **No ML Classifiers** - Simple, reliable approach that avoids complex custom models

## Quick Start

### 1. Install Dependencies

```bash
cd voicebeat
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Also ensure FFmpeg is installed:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt-get install ffmpeg
```

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp config/.env.example .env
```

Edit `.env` with your API keys:
- `SMALLEST_API_KEY` - Get from https://console.smallest.ai/apikeys
- `OPENAI_API_KEY` - Get from https://platform.openai.com/api-keys

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### 4. Explore the API

Open http://localhost:8000/docs for interactive Swagger documentation.

## Core Workflow

### Single Recording (Recommended)

1. **Record everything in one take**: Speak instructions, hum melodies, beatbox rhythms
2. **Upload via `/api/v1/process`**: System automatically segments, analyzes, and orchestrates
3. **Get your track**: Receive the mixed output with all layers combined

Example recording:
> "Give me a piano melody" *[hum melody]* "Now add some drums" *[beatbox rhythm]*

### Simple Mode (Separate Files)

For testing individual components:
- `POST /api/v1/describe` - Transcribe speech and parse instructions
- `POST /api/v1/rhythm` - Analyze rhythm from beatboxing/humming

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/process` | **Main endpoint** - Process single recording (speech + music) |
| `POST` | `/api/v1/describe` | Simple mode: Parse spoken description |
| `POST` | `/api/v1/rhythm` | Simple mode: Analyze rhythm audio |
| `POST` | `/api/v1/projects` | Create a new project |
| `GET` | `/api/v1/projects/{id}` | Get project details |
| `POST` | `/api/v1/projects/{id}/layers` | Add a layer from recording |
| `DELETE` | `/api/v1/projects/{id}/layers/{layer_id}` | Remove a layer |
| `GET` | `/api/v1/projects/{id}/mix` | Mix all layers into MP3 |
| `POST` | `/api/v1/speak` | Text-to-speech feedback |
| `GET` | `/samples/catalog` | List available samples |
| `GET` | `/download/{filename}` | Download generated audio |
| `GET` | `/health` | Health check |

## Architecture

```
Single Recording (talk + hum + beatbox)
                │
                ▼
┌──────────────────────────────────────┐
│  SEGMENTATION (segmenter.py)         │
│  STT timestamps → speech regions     │
│  Gaps → music segments               │
│  Classify: melody vs rhythm          │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  ANALYSIS                            │
│  Speech → transcription → parsed     │
│  Melody → pitch detection → ABC      │
│  Rhythm → onset detection → grid     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  AI ORCHESTRATOR (agent.py)          │
│  OpenAI with tool use:               │
│  - assign_instrument()               │
│  - set_tempo(), set_genre()          │
│  - render_rhythm_layer()             │
│  - render_melody_layer()             │
│  - combine_layers()                  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  ASSEMBLY (track_assembler.py)       │
│  Place samples on rhythm grid        │
│  Synthesize melodies                 │
│  Mix layers → normalize → export     │
└──────────────────────────────────────┘
```

## Project Structure

```
voicebeat/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── api/routes.py              # API endpoints
│   ├── models/schemas.py          # Pydantic models
│   ├── services/
│   │   ├── transcription.py       # Smallest.ai Pulse STT
│   │   ├── tts.py                 # Smallest.ai Waves TTS
│   │   ├── segmenter.py           # Audio segmentation (key innovation)
│   │   ├── pitch_analyzer.py      # Melody extraction (librosa.pyin)
│   │   ├── rhythm_analyzer.py     # Beat detection (librosa)
│   │   ├── agent.py               # OpenAI orchestrator with tools
│   │   ├── description_parser.py  # Instruction parsing (OpenAI)
│   │   ├── sample_lookup.py       # Local sample file mapping (no R2)
│   │   ├── track_assembler.py     # pydub audio assembly
│   │   └── notation.py            # ABC notation builder
│   └── utils/
│       ├── audio.py               # Audio helpers
│       └── music_theory.py        # Key detection, quantization
├── config/
│   ├── settings.py                # Configuration
│   └── .env.example               # Environment template
├── samples/                        # Audio sample library (local only)
├── output/                         # Generated audio files
├── cache/                          # R2 cache directory (auto-created)
├── scripts/
│   ├── generate_samples.py        # Create placeholder samples
│   ├── build_catalog.py           # Build local sample catalog
│   └── setup_r2.py                # Configure R2 for user content
├── tests/                          # Test suite
└── requirements.txt
```

## Storage Architecture

VoiceBeat uses a hybrid storage approach:

### Local Storage (Always)
- **Sample library**: `samples/` directory - never uses cloud storage
- **Sample catalog**: `catalog/samples.json` - generated from local files
- **Development output**: `output/` directory for local testing

### Cloud Storage (Optional - R2)
- **User recordings**: Speech segments from voice input
- **Generated layers**: Individual instrument tracks  
- **Mixed outputs**: Final project audio files
- **Cache management**: Local cache with TTL for downloaded files

### Configuration
```bash
# Required for R2 features (user content only)
USE_R2_STORAGE=true
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=your_bucket_name

# Sample files are always local regardless of R2 settings
SAMPLES_DIR=./samples
```

## Tech Stack

- **Backend**: FastAPI (async)
- **Speech-to-text**: Smallest.ai Pulse
- **Text-to-speech**: Smallest.ai Waves Lightning
- **AI Orchestration**: OpenAI GPT-4o with function calling
- **Audio Analysis**: librosa (pitch detection, onset detection)
- **Audio Assembly**: pydub + FFmpeg
- **Musical Notation**: ABC notation as intermediate format

## Running Tests

```bash
pytest tests/
```

## How This Differs from Similar Projects

| Challenge | Traditional Approach | VoiceBeat |
|---|---|---|
| Audio segmentation | Custom ML classifier | STT word timestamps + energy detection |
| Audio → musical data | Audio-to-MIDI (unreliable) | librosa pitch detection → ABC notation |
| Instruction execution | Direct LLM generation (hallucination) | Agentic tool use (no hallucination) |
| Input requirements | Separate files | Single recording |
