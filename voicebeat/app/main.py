import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.api.routes import router
from config.settings import settings

# Configure logging to show all our debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
# Set our app modules to INFO level
for module in ['app.api.routes', 'app.services.segmenter', 'app.services.transcription', 'app.services.agent']:
    logging.getLogger(module).setLevel(logging.INFO)


app = FastAPI(
    title="VoiceBeat API",
    description="Voice-driven music creation tool",
    version="0.1.0",
)

# CORS middleware - wide open for MVP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main interface."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "VoiceBeat API - visit /docs for API documentation"}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Validate required API keys
    if not settings.smallest_api_key:
        logging.error("=" * 60)
        logging.error("MISSING API KEY: SMALLEST_API_KEY is not set!")
        logging.error("Please add it to your .env file:")
        logging.error("  SMALLEST_API_KEY=your_api_key_here")
        logging.error("Get your key from https://smallest.ai")
        logging.error("=" * 60)

    if not settings.openai_api_key:
        logging.error("=" * 60)
        logging.error("MISSING API KEY: OPENAI_API_KEY is not set!")
        logging.error("Please add it to your .env file:")
        logging.error("  OPENAI_API_KEY=sk-xxxxx")
        logging.error("=" * 60)

    # Ensure output directory exists
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure samples directory exists
    settings.samples_dir.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
