"""
API Routes for VoiceBeat

The main endpoint is /api/v1/process which accepts a single recording
containing both speech (instructions) and music (humming/beatboxing).
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import boto3
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, Response

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

from config.settings import settings

from app.models.schemas import (
    DescribeResponse,
    HealthResponse,
    Layer,
    MusicDescription,
    ProcessResponse,
    Project,
    ProjectCreateRequest,
    RhythmResponse,
    SampleCatalogResponse,
    SegmentType,
    SpeakRequest,
)
from app.services import (
    agent,
    description_parser,
    rhythm_analyzer,
    sample_lookup,
    segmenter,
    track_assembler,
    transcription,
    tts,
    visualizer,
)
from app.utils.audio import get_content_type, read_upload_file, validate_audio_file

router = APIRouter()


@router.get("/api/docs", include_in_schema=False)
async def swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="VoiceBeat API - Swagger UI",
    )


@router.get("/api/redoc", include_in_schema=False)
async def redoc_ui():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="VoiceBeat API - ReDoc",
    )


# In-memory project storage
projects: dict[str, Project] = {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/api/v1/process", response_model=ProcessResponse)
async def process_recording(
    audio: UploadFile = File(...),
    project_name: str = "New Project",
):
    """
    Main endpoint: Process a single audio recording.

    Accepts a recording where the user both talks (instructions) and
    hums/beatboxes (musical content) in one take.

    Pipeline:
    1. Segment audio into speech vs music regions using STT timestamps
    2. Transcribe speech, analyze musical segments
    3. Use AI agent to orchestrate: interpret instructions, assign instruments
    4. Render layers and combine into final output

    Returns the project with all layers and feedback text for TTS.
    """
    logger.info("=" * 80)
    logger.info("PROCESS: New recording received")
    logger.info(f"PROCESS: Filename: {audio.filename}")
    logger.info(f"PROCESS: Content-Type header: {audio.content_type}")
    logger.info(f"PROCESS: Project name: {project_name}")

    if not await validate_audio_file(audio):
        logger.error("PROCESS: Invalid audio file format")
        raise HTTPException(status_code=400, detail="Invalid audio file format")

    audio_bytes = await read_upload_file(audio)
    content_type = get_content_type(audio.filename or "audio.wav")
    logger.info(
        f"PROCESS: Read {len(audio_bytes)} bytes, determined content_type={content_type}"
    )

    # Step 1: Segment the recording
    logger.info("PROCESS: Step 1 - Segmenting recording...")
    segments = await segmenter.segment_recording(audio_bytes, content_type)

    if not segments:
        logger.error("PROCESS: No segments detected")
        raise HTTPException(status_code=400, detail="No segments detected in audio")

    logger.info(f"PROCESS: Found {len(segments)} segments")

    # Step 2: Extract audio for each segment
    logger.info("PROCESS: Step 2 - Extracting audio for each segment...")
    segment_audio_data: dict[str, bytes] = {}
    for seg in segments:
        if seg.type != SegmentType.SPEECH:
            seg_audio_path = await segmenter.extract_segment_audio(audio_bytes, seg)
            with open(seg_audio_path, "rb") as f:
                segment_audio_data[seg.id] = f.read()
            seg.audio_file = seg_audio_path
            logger.info(
                f"PROCESS: Extracted segment {seg.id[:8]} -> {seg_audio_path} ({len(segment_audio_data[seg.id])} bytes)"
            )

    # Step 3: Get speech transcripts for instructions
    logger.info("PROCESS: Step 3 - Extracting speech transcripts...")
    speech_transcripts = [
        seg.transcript
        for seg in segments
        if seg.type == SegmentType.SPEECH and seg.transcript
    ]
    logger.info(
        f"PROCESS: Found {len(speech_transcripts)} speech transcripts: {speech_transcripts}"
    )

    # Create project
    project_id = str(uuid.uuid4())
    project = Project(
        id=project_id,
        name=project_name,
        segments=segments,
    )
    logger.info(f"PROCESS: Created project {project_id}")

    # Step 4: If we have speech instructions, parse them
    if speech_transcripts:
        logger.info("PROCESS: Step 4 - Parsing speech instructions...")
        description = await description_parser.extract_instructions(speech_transcripts)
        project.description = description
        if description.tempo_bpm:
            project.bpm = description.tempo_bpm
        logger.info(
            f"PROCESS: Parsed description: genre={description.genre}, tempo={description.tempo_bpm}, instruments={description.instruments}"
        )
    else:
        logger.info("PROCESS: Step 4 - No speech instructions to parse")

    # Step 5: Run the AI agent to orchestrate
    logger.info("PROCESS: Step 5 - Running AI agent...")
    project, summary = await agent.orchestrate(
        segments=segments,
        speech_instructions=speech_transcripts,
        segment_audio_data=segment_audio_data,
        project=project,
    )
    logger.info(f"PROCESS: Agent summary: {summary}")
    logger.info(f"PROCESS: Project now has {len(project.layers)} layers")

    # Step 6: Mix if we have layers
    if project.layers:
        logger.info("PROCESS: Step 6 - Mixing layers...")
        try:
            project.mixed_file = track_assembler.mix_project(project)
            logger.info(f"PROCESS: Mixed file: {project.mixed_file}")
        except ValueError as e:
            logger.warning(f"PROCESS: Mix failed: {e}")

    # Save project
    projects[project_id] = project

    logger.info("PROCESS: Complete!")
    logger.info("=" * 80)

    return ProcessResponse(
        project=project,
        feedback_text=summary,
    )


@router.post("/api/v1/visualize")
async def visualize_audio(audio: UploadFile = File(...)):
    """
    Analyze audio and return visualization data.

    Returns waveform, onset detection, beat tracking, and pitch analysis
    for debugging and visual feedback.
    """
    logger.info("=" * 80)
    logger.info("VISUALIZE: New audio received for visualization")
    logger.info(f"VISUALIZE: Filename: {audio.filename}")

    if not await validate_audio_file(audio):
        logger.error("VISUALIZE: Invalid audio file format")
        raise HTTPException(status_code=400, detail="Invalid audio file format")

    audio_bytes = await read_upload_file(audio)
    logger.info(f"VISUALIZE: Read {len(audio_bytes)} bytes")

    try:
        viz_data = await visualizer.analyze_for_visualization(audio_bytes)

        return {
            "waveform": viz_data.waveform,
            "waveform_times": viz_data.waveform_times,
            "duration_seconds": viz_data.duration_seconds,
            "sample_rate": viz_data.sample_rate,
            "onset_times": viz_data.onset_times,
            "onset_strengths": viz_data.onset_strengths,
            "beat_times": viz_data.beat_times,
            "tempo_bpm": viz_data.tempo_bpm,
            "onset_envelope": viz_data.onset_envelope,
            "onset_envelope_times": viz_data.onset_envelope_times,
            "pitch_times": viz_data.pitch_times,
            "pitch_frequencies": viz_data.pitch_frequencies,
            "pitch_confidences": viz_data.pitch_confidences,
            "rms_values": viz_data.rms_values,
            "rms_times": viz_data.rms_times,
        }
    except Exception as e:
        logger.error(f"VISUALIZE: Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.post("/api/v1/describe", response_model=DescribeResponse)
async def describe_audio(audio: UploadFile = File(...)):
    """
    Simple mode: Transcribe audio and parse music description.

    Accepts an audio file of spoken description only,
    transcribes it and extracts structured metadata.
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
async def analyze_rhythm_endpoint(
    audio: UploadFile = File(...),
    target_bpm: int | None = None,
):
    """
    Simple mode: Analyze rhythm from audio.

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
    """Create a new empty project."""
    project_id = str(uuid.uuid4())
    project = Project(
        id=project_id,
        name=request.name,
        bpm=request.bpm,
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
    audio: UploadFile = File(...),
):
    """
    Add a new layer to a project from a single recording.

    The recording is segmented and processed through the AI agent.
    """
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]

    if not await validate_audio_file(audio):
        raise HTTPException(status_code=400, detail="Invalid audio format")

    audio_bytes = await read_upload_file(audio)
    content_type = get_content_type(audio.filename or "audio.wav")

    # Segment the recording
    segments = await segmenter.segment_recording(audio_bytes, content_type)

    # Extract segment audio
    segment_audio_data: dict[str, bytes] = {}
    for seg in segments:
        if seg.type != SegmentType.SPEECH:
            seg_audio_path = await segmenter.extract_segment_audio(audio_bytes, seg)
            with open(seg_audio_path, "rb") as f:
                segment_audio_data[seg.id] = f.read()
            seg.audio_file = seg_audio_path

    # Get speech transcripts
    speech_transcripts = [
        seg.transcript
        for seg in segments
        if seg.type == SegmentType.SPEECH and seg.transcript
    ]

    # Add segments to project
    project.segments.extend(segments)

    # Run agent
    project, _ = await agent.orchestrate(
        segments=segments,
        speech_instructions=speech_transcripts,
        segment_audio_data=segment_audio_data,
        project=project,
    )

    project.updated_at = datetime.now()

    if not project.layers:
        raise HTTPException(status_code=500, detail="Failed to create layer")

    return project.layers[-1]


@router.delete("/api/v1/projects/{project_id}/layers/{layer_id}")
async def delete_layer(project_id: str, layer_id: str):
    """Remove a layer from a project."""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    original_count = len(project.layers)
    project.layers = [l for l in project.layers if l.id != layer_id]

    if len(project.layers) == original_count:
        raise HTTPException(status_code=404, detail="Layer not found")

    project.updated_at = datetime.now()
    return {"message": "Layer deleted"}


@router.get("/api/v1/projects/{project_id}/mix")
async def mix_project_endpoint(project_id: str):
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
        project.updated_at = datetime.now()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return FileResponse(
        path=output_path,
        media_type="audio/mpeg",
        filename=Path(output_path).name,
    )


@router.post(
    "/api/v1/tts",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def smallest_tts(request: SpeakRequest):
    """
    Convert text to speech using Smallest.ai Waves Lightning TTS.

    Returns MP3 audio bytes.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    audio_bytes = await tts.speak_mp3(request.text, request.voice_id)

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="speech.mp3"'},
    )


@router.post(
    "/api/v1/speak",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def speak(request: SpeakRequest):
    """
    Convert text to speech using Waves TTS.

    Returns WAV audio bytes for voice feedback.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    audio_bytes = await tts.speak(request.text, request.voice_id)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@router.post("/api/v1/r2/upload")
async def upload_to_r2(
    file: UploadFile = File(...),
    r2_bucket_name: str | None = None,
    r2_endpoint: str | None = None,
    r2_public: str | None = None,
):
    """
    Upload a file to R2 and return a presigned download URL.
    Uses endpoint, bucket name, and public URL from environment settings.
    """
    bucket = r2_bucket_name or settings.r2_bucket_name
    public_url = r2_public or settings.r2_public_url
    endpoint = r2_endpoint

    if not endpoint:
        if not settings.r2_account_id:
            raise HTTPException(status_code=500, detail="R2 is not configured")
        endpoint = f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"

    if not all([settings.r2_access_key_id, settings.r2_secret_access_key, bucket]):
        raise HTTPException(status_code=500, detail="R2 is not configured")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    content_type = get_content_type(file.filename)
    safe_name = Path(file.filename).name
    object_key = f"uploads/{uuid.uuid4()}-{safe_name}"

    r2_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
    )

    r2_client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=file_bytes,
        ContentType=content_type,
    )

    expires_in_seconds = 3600
    download_url = r2_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=expires_in_seconds,
    )

    public_download_url = None
    if public_url:
        public_download_url = f"{public_url.rstrip('/')}/{object_key}"

    return {
        "key": object_key,
        "download_url": download_url,
        "public_download_url": public_download_url,
        "expires_in": expires_in_seconds,
    }


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
