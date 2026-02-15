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

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
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

# In-memory project storage
projects: dict[str, Project] = {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/api/v1/process", response_model=ProcessResponse)
async def process_recording(
    request: Request,
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
    storage = request.app.state.storage
    segment_audio_data: dict[str, bytes] = {}
    for seg in segments:
        seg_audio_url = await segmenter.extract_segment_audio(audio_bytes, seg, storage)
        # For processing, we need the raw bytes - get from cache
        try:
            r2_key = f"segments/{seg.id}.wav"
            cached_path = await storage.get_file(r2_key)
            with open(cached_path, "rb") as f:
                segment_audio_data[seg.id] = f.read()
        except Exception as e:
            logger.error(f"Failed to read segment audio: {e}")
            continue
        seg.audio_file = seg_audio_url
        logger.info(
            f"PROCESS: Extracted segment {seg.id[:8]} ({seg.type.value}) -> {seg_audio_url} ({len(segment_audio_data[seg.id])} bytes)"
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
            project.mixed_file = await track_assembler.mix_project(project, storage)
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


@router.post("/api/v1/projects/{project_id}/refine", response_model=ProcessResponse)
async def refine_project(
    project_id: str,
    instructions: str = Form(""),
    audio: UploadFile = File(None),
):
    """
    Refine an existing project with text instructions and/or new audio.

    Supports three modes:
    1. Text-only: "add bass drums", "remove the piano"
    2. Audio + text: user hums/beatboxes new content with instructions
    3. Vocal: user sings and says "add my voice" -> autotuned vocal layer
    """
    logger.info("=" * 80)
    logger.info(f"REFINE: Project {project_id}")
    logger.info(f"REFINE: Instructions: {instructions}")
    logger.info(
        f"REFINE: Audio provided: {audio is not None and audio.filename is not None}"
    )

    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    segments = None
    segment_audio_data = None

    # If audio is provided, segment and analyze it
    if audio and audio.filename:
        from app.utils.audio import (
            get_content_type,
            read_upload_file,
            validate_audio_file,
        )

        if not await validate_audio_file(audio):
            raise HTTPException(status_code=400, detail="Invalid audio file format")

        audio_bytes = await read_upload_file(audio)
        content_type = get_content_type(audio.filename or "audio.wav")

        logger.info(f"REFINE: Segmenting audio ({len(audio_bytes)} bytes)")
        segments = await segmenter.segment_recording(audio_bytes, content_type)

        # Extract audio for non-speech segments
        segment_audio_data = {}
        for seg in segments:
            if seg.type != SegmentType.SPEECH:
                seg_audio_path = await segmenter.extract_segment_audio(audio_bytes, seg)
                with open(seg_audio_path, "rb") as f:
                    segment_audio_data[seg.id] = f.read()
                seg.audio_file = seg_audio_path

        # Collect speech transcripts and append to instructions
        speech_transcripts = [
            seg.transcript
            for seg in segments
            if seg.type == SegmentType.SPEECH and seg.transcript
        ]
        if speech_transcripts:
            instructions = instructions + " " + " ".join(speech_transcripts)
            logger.info(f"REFINE: Combined instructions: {instructions}")

        # Add new segments to project
        project.segments.extend(segments)

    # Run refinement orchestration
    logger.info("REFINE: Running AI agent for refinement...")
    project, summary = await agent.orchestrate_refinement(
        project=project,
        instructions=instructions,
        segments=segments,
        segment_audio_data=segment_audio_data,
    )

    # Re-mix if we have layers
    if project.layers:
        try:
            project.mixed_file = track_assembler.mix_project(project)
            logger.info(f"REFINE: Mixed file: {project.mixed_file}")
        except ValueError as e:
            logger.warning(f"REFINE: Mix failed: {e}")

    # Save updated project
    projects[project_id] = project

    logger.info("REFINE: Complete!")
    logger.info("=" * 80)

    return ProcessResponse(
        project=project,
        feedback_text=summary,
    )


@router.post("/api/v1/projects/{project_id}/mix")
async def mix_project_endpoint(request: Request, project_id: str):
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
        storage = request.app.state.storage
        output_url = await track_assembler.mix_project(project, storage)
        project.mixed_file = output_url
        project.updated_at = datetime.now()

        # For file response, we need to get the actual file path from cache
        r2_key = (
            output_url.split("/")[-1] if output_url.startswith("http") else output_url
        )
        cached_path = await storage.get_file(r2_key)

        return FileResponse(
            path=str(cached_path),
            media_type="audio/mpeg",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mix project: {str(e)}")


@router.post("/api/v1/speak")
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


@router.get("/api/v1/samples", response_model=SampleCatalogResponse)
async def get_sample_catalog(request: Request):
    """List all available samples in the library."""
    catalog = sample_lookup.get_sample_catalog()
    return SampleCatalogResponse(**catalog)


@router.get("/api/v1/logs")
async def get_logs(since: int = 0):
    """Get recent logs. 'since' = index to start from (for polling)."""
    from app.main import log_buffer

    logs = log_buffer.get_logs()
    return {"logs": logs[since:], "total": len(logs)}


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
