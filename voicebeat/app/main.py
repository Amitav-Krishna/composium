import asyncio
import collections
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings

from app.api.routes import router
from app.services.file_storage import FileStorageService
from app.services.sample_lookup import SampleLookup


class MemoryLogHandler(logging.Handler):
    """In-memory ring buffer that captures recent log entries."""

    def __init__(self, capacity=500):
        super().__init__()
        self.buffer = collections.deque(maxlen=capacity)

    def emit(self, record):
        self.buffer.append(self.format(record))

    def get_logs(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()


# Configure logging to show all our debug info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Set our app modules to INFO level
for module in [
    "app.api.routes",
    "app.services.segmenter",
    "app.services.transcription",
    "app.services.agent",
]:
    logging.getLogger(module).setLevel(logging.INFO)

# Add in-memory log buffer for the log viewer
log_buffer = MemoryLogHandler(capacity=500)
log_buffer.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(log_buffer)


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

    # Validate R2 configuration if R2 storage is enabled
    if settings.use_r2_storage:
        if not all(
            [
                settings.r2_account_id,
                settings.r2_access_key_id,
                settings.r2_secret_access_key,
            ]
        ):
            logging.warning("=" * 60)
            logging.warning("R2 STORAGE ENABLED BUT MISSING CONFIGURATION!")
            logging.warning("Please add to your .env file:")
            logging.warning("  R2_ACCOUNT_ID=your_account_id")
            logging.warning("  R2_ACCESS_KEY_ID=your_access_key")
            logging.warning("  R2_SECRET_ACCESS_KEY=your_secret_key")
            logging.warning("Falling back to local storage...")
            logging.warning("=" * 60)
            settings.use_r2_storage = False

    # Initialize file storage
    app.state.storage = FileStorageService(
        cache_dir=settings.cache_dir, cache_ttl_hours=settings.cache_ttl_hours
    )
    if settings.use_r2_storage:
        await app.state.storage.start_background_uploader()

    # Initialize sample lookup
    app.state.samples = SampleLookup()
    app.state.samples.initialize()

    # Start cache cleanup task
    asyncio.create_task(periodic_cache_cleanup())

    # Ensure output directory exists (for fallback)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure samples directory exists (for local fallback)
    settings.samples_dir.mkdir(parents=True, exist_ok=True)


async def periodic_cache_cleanup():
    """Clean cache every hour"""
    while True:
        await asyncio.sleep(3600)  # 1 hour
        if hasattr(app.state, "storage"):
            app.state.storage.cleanup_cache()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
