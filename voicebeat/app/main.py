from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.api.routes import router
from config.settings import settings


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


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
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
