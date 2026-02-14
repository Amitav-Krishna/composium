import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from config.settings import settings
from app.models.schemas import (
    DescribeResponse,
    RhythmResponse,
    Project,
    Layer,
    ProjectCreateRequest,
    SpeakRequest,
    HealthResponse,
    SampleCatalogResponse,
)
from app.services import transcription, tts, description_parser, rhythm_analyzer, sample_lookup, track_assembler
from app.utils.audio import get_content_type, validate_audio_file, read_upload_file


router = APIRouter()

# In-memory project storage
projects: dict[str, Project] = {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/api/v1/describe", response_model=DescribeResponse)
async def describe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio and parse music description.

    Accepts an audio file, transcribes it using Pulse STT,
    then parses the transcript using Claude to extract structured metadata.
    """
    if not await validate_audio_file(audio):
        raise HTTPException(status_code=400, detail="Invalid audio file format")

    audio_bytes = await read_upload_file(audio)
    content_type = get_content_type(audio.filename or "audio.wav")

    # Transcribe
    transcript = await transcription.transcribe(audio_bytes, content_type)

    if not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    # Parse description
    description = await description_parser.parse_description(transcript)

    return DescribeResponse(transcript=transcript, description=description)


@router.post("/api/v1/rhythm", response_model=RhythmResponse)
async def analyze_rhythm(
    audio: UploadFile = File(...),
    target_bpm: int | None = None,
):
    """
    Analyze rhythm from audio.

    Accepts an audio file (humming/beatboxing) and detects onsets,
    quantizing them to a 16th-note grid.
    """
    if not await validate_audio_file(audio):
        raise HTTPException(status_code=400, detail="Invalid audio file format")

    audio_bytes = await read_upload_file(audio)

    rhythm, onsets = await rhythm_analyzer.analyze_rhythm(audio_bytes, target_bpm)

    return RhythmResponse(rhythm=rhythm, detected_onsets=onsets)


@router.post("/api/v1/projects", response_model=Project)
async def create_project(request: ProjectCreateRequest):
    """Create a new project."""
    project_id = str(uuid.uuid4())
    project = Project(
        id=project_id,
        name=request.name,
        bpm=request.bpm,
        layers=[],
    )
    projects[project_id] = project
    return project


@router.get("/api/v1/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get project details."""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects[project_id]


@router.post("/api/v1/projects/{project_id}/layers", response_model=Layer)
async def add_layer(
    project_id: str,
    description_audio: UploadFile = File(...),
    rhythm_audio: UploadFile = File(...),
):
    """
    Add a new layer to a project.

    Accepts two audio files:
    - description_audio: Voice describing what kind of music
    - rhythm_audio: Humming/beatboxing the rhythm

    Runs the full pipeline: transcribe -> parse -> detect rhythm -> lookup samples -> render
    """
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]

    # Validate audio files
    if not await validate_audio_file(description_audio):
        raise HTTPException(status_code=400, detail="Invalid description audio format")
    if not await validate_audio_file(rhythm_audio):
        raise HTTPException(status_code=400, detail="Invalid rhythm audio format")

    # Read audio files
    desc_bytes = await read_upload_file(description_audio)
    rhythm_bytes = await read_upload_file(rhythm_audio)

    # Step 1: Transcribe description
    desc_content_type = get_content_type(description_audio.filename or "audio.wav")
    transcript = await transcription.transcribe(desc_bytes, desc_content_type)

    if not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe description audio")

    # Step 2: Parse description
    description = await description_parser.parse_description(transcript)

    # Step 3: Analyze rhythm
    rhythm, _ = await rhythm_analyzer.analyze_rhythm(rhythm_bytes, project.bpm)

    # Step 4: Assign instruments to beats
    rhythm = rhythm_analyzer.assign_instruments(rhythm, description.instruments)

    # Step 5: Lookup samples
    sample_mapping = sample_lookup.lookup_samples(description)

    if not sample_mapping:
        raise HTTPException(status_code=500, detail="No samples found for the requested instruments")

    # Step 6: Render layer
    audio_file = track_assembler.render_layer(
        rhythm=rhythm,
        sample_mapping=sample_mapping,
        bpm=project.bpm,
    )

    # Create layer
    layer = Layer(
        id=str(uuid.uuid4()),
        description=description,
        rhythm=rhythm,
        sample_mapping=sample_mapping,
        audio_file=audio_file,
    )

    # Add to project
    project.layers.append(layer)

    return layer


@router.get("/api/v1/projects/{project_id}/mix")
async def mix_project(project_id: str):
    """
    Mix all layers in a project into a final MP3.

    Returns the mixed audio file.
    """
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]

    if not project.layers:
        raise HTTPException(status_code=400, detail="Project has no layers to mix")

    try:
        output_path = track_assembler.mix_project(project)
        project.mixed_file = output_path
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return FileResponse(
        path=output_path,
        media_type="audio/mpeg",
        filename=Path(output_path).name,
    )


@router.post("/api/v1/speak")
async def speak(request: SpeakRequest):
    """
    Convert text to speech using Waves TTS.

    Returns WAV audio bytes.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    audio_bytes = await tts.speak(request.text, request.voice_id)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="speech.wav"'},
    )


@router.get("/samples/catalog", response_model=SampleCatalogResponse)
async def get_sample_catalog():
    """List all available samples in the library."""
    catalog = sample_lookup.get_sample_catalog()
    return SampleCatalogResponse(**catalog)


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated audio file."""
    output_dir = Path(settings.output_dir)
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Prevent directory traversal
    if not file_path.resolve().is_relative_to(output_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine media type
    ext = file_path.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )
