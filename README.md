# VibeBeat

Record yourself talking and humming into a microphone, and VibeBeat turns it into a produced music track. One recording, one click, real instruments.

## What it does

You hit record, say something like *"give me a chill hip-hop beat with piano"*, hum a melody or beatbox a rhythm, and stop. VibeBeat takes that single recording and:

1. **Separates speech from music** — uses speech-to-text timestamps to figure out which parts are instructions and which parts are you performing
2. **Understands your instructions** — an AI agent interprets what you asked for (genre, instruments, tempo, mood)
3. **Analyzes your performance** — detects pitch, rhythm, and beat patterns from your humming/beatboxing using librosa
4. **Generates a vocal track** — synthesizes sung vocals with TTS, pitch-shifts them to target notes, and time-stretches them to land on beat positions
5. **Renders real instruments** — converts your input into MIDI notation and renders through Composium's instrument pipeline
6. **Mixes everything** — layers vocals, melody, rhythm, and instruments into a final MP3 you can export

## Why

Making music usually requires knowing an instrument, a DAW, or music theory. VibeBeat lets anyone sketch a song idea with just their voice. The gap between *having a melody in your head* and *hearing it as a produced track* should be one recording, not hours of production.

## Prerequisites

- **Python 3.11+**
- **FFmpeg** — required for audio processing
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```
- **API keys** (add these to a `.env` file in the project root):

  | Key | What it's for | Where to get it |
  |-----|---------------|-----------------|
  | `OPENAI_API_KEY` | AI orchestration (GPT-4o) | https://platform.openai.com/api-keys |
  | `ELEVENLABS_API_KEY` | Text-to-speech vocals | https://elevenlabs.io |
  | `OPENROUTER_API_KEY` | Subagent LLM calls (cheaper models) | https://openrouter.ai/keys |

## Getting started

```bash
cd voicebeat
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your API keys:

```
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
```

Start the server:

```bash
cd voicebeat
uvicorn app.main:app --reload
```

Open http://localhost:8000 in your browser. Hit the record button, speak and hum, and wait for your track.

## Optional: Voice cloning with RVC

If you want the generated vocals to sound like a specific voice, you can provide an RVC model:

```
RVC_MODEL_PATH=/path/to/model.pth
RVC_INDEX_PATH=/path/to/model.index
```

Leave these empty to use the default ElevenLabs voice.
