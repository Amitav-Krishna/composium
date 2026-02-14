# VoiceBeat

Voice-driven music creation tool. Users speak to describe what kind of music they want, hum/beatbox the rhythm, and the app assembles layered tracks from audio samples.

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
- `ANTHROPIC_API_KEY` - Get from https://console.anthropic.com/

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### 4. Explore the API

Open http://localhost:8000/docs for interactive Swagger documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/describe` | Transcribe audio + parse music description |
| `POST` | `/api/v1/rhythm` | Analyze rhythm from audio |
| `POST` | `/api/v1/projects` | Create a new project |
| `GET` | `/api/v1/projects/{id}` | Get project details |
| `POST` | `/api/v1/projects/{id}/layers` | Add a layer (full pipeline) |
| `GET` | `/api/v1/projects/{id}/mix` | Mix all layers into MP3 |
| `POST` | `/api/v1/speak` | Text-to-speech |
| `GET` | `/samples/catalog` | List available samples |
| `GET` | `/download/{filename}` | Download generated audio |

## Core Workflow

1. **Create a project**: `POST /api/v1/projects`
2. **Add layers**: `POST /api/v1/projects/{id}/layers` with:
   - `description_audio`: Voice describing the music style
   - `rhythm_audio`: Humming/beatboxing the rhythm
3. **Mix the project**: `GET /api/v1/projects/{id}/mix` to get the final MP3

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Project Structure

```
voicebeat/
├── app/
│   ├── main.py              # FastAPI app
│   ├── api/routes.py        # API endpoints
│   ├── models/schemas.py    # Pydantic models
│   ├── services/            # Core services
│   │   ├── transcription.py # Smallest.ai Pulse STT
│   │   ├── tts.py           # Smallest.ai Waves TTS
│   │   ├── description_parser.py  # Claude API
│   │   ├── rhythm_analyzer.py     # librosa
│   │   ├── sample_lookup.py       # Sample file mapping
│   │   └── track_assembler.py     # pydub audio assembly
│   └── utils/audio.py       # Audio helpers
├── config/
│   ├── settings.py          # Configuration
│   └── .env.example         # Environment template
├── samples/                  # Audio sample library
├── output/                   # Generated audio files
├── tests/                    # Test suite
└── requirements.txt
```

## Tech Stack

- **Backend**: FastAPI
- **Speech-to-text**: Smallest.ai Pulse
- **Text-to-speech**: Smallest.ai Waves Lightning
- **Description parsing**: Anthropic Claude
- **Rhythm detection**: librosa
- **Audio assembly**: pydub + FFmpeg
