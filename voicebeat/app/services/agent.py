"""
Agentic Orchestrator Service — thin shim.

Delegates to the Director + Subagent architecture while preserving the
exact orchestrate() and orchestrate_refinement() signatures so that
routes.py requires zero changes.
"""

import logging
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

from config.settings import settings
from app.models.schemas import (
    AudioSegment, SegmentType, Project,
)
from app.services import rhythm_analyzer, pitch_analyzer


# ───────────────────────────────────────────────────────────────────────
# Pre-analysis (shared between orchestrate and orchestrate_refinement)
# ───────────────────────────────────────────────────────────────────────

async def _pre_analyze(
    segments: list[AudioSegment],
    segment_audio_data: dict[str, bytes],
) -> dict[str, dict]:
    """Analyze musical segments (rhythm + melody) and return segment data dict."""
    segment_data: dict[str, dict] = {}

    for segment in segments:
        if segment.type == SegmentType.SPEECH:
            continue
        if segment.id not in segment_audio_data:
            continue

        audio_bytes = segment_audio_data[segment.id]
        logger.info(
            f"AGENT: Analyzing segment {segment.id[:8]} "
            f"({segment.type.value}, {len(audio_bytes)} bytes)"
        )

        if segment.type == SegmentType.RHYTHM:
            rhythm, onsets = await rhythm_analyzer.analyze_rhythm(audio_bytes)
            segment_data[segment.id] = {
                "type": "rhythm",
                "rhythm": rhythm,
                "onsets": onsets,
            }
            logger.info(f"AGENT:   -> Rhythm: {len(rhythm.beats)} beats, {rhythm.bpm} BPM")

        elif segment.type == SegmentType.MELODY:
            melody = await pitch_analyzer.analyze_melody(audio_bytes)
            segment_data[segment.id] = {
                "type": "melody",
                "melody": melody,
            }
            logger.info(f"AGENT:   -> Melody: {len(melody.pitches)} pitches")

    return segment_data


# ───────────────────────────────────────────────────────────────────────
# Public API (signatures unchanged from the monolithic agent)
# ───────────────────────────────────────────────────────────────────────

async def orchestrate(
    segments: list[AudioSegment],
    speech_instructions: list[str],
    segment_audio_data: dict[str, bytes],
    project: Project,
) -> tuple[Project, str]:
    """
    Orchestrate music creation via the Director + Subagent architecture.

    Args:
        segments: Classified audio segments
        speech_instructions: Extracted speech transcripts
        segment_audio_data: segment_id -> audio bytes
        project: The project being built

    Returns:
        (updated project, summary text for TTS feedback)
    """
    logger.info("=" * 60)
    logger.info("AGENT: Starting orchestration (director mode)")
    logger.info(f"AGENT: Segments: {len(segments)}")
    logger.info(f"AGENT: Speech: {speech_instructions}")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set! Add it to your .env file.")

    # Pre-analyze musical segments
    segment_data = await _pre_analyze(segments, segment_audio_data)

    # Run director
    from app.services.director import Director

    director = Director()
    summary = await director.run(
        project, segments, speech_instructions,
        segment_data, segment_audio_data,
    )

    project.updated_at = datetime.now()

    logger.info(f"AGENT: Final summary: {summary}")
    logger.info(f"AGENT: Project has {len(project.layers)} total layers")
    logger.info("=" * 60)

    return project, summary


async def orchestrate_refinement(
    project: Project,
    instructions: str,
    segments: list[AudioSegment] | None = None,
    segment_audio_data: dict[str, bytes] | None = None,
) -> tuple[Project, str]:
    """
    Refine an existing project via the Director + Subagent architecture.

    Args:
        project: Existing project to refine
        instructions: Text instructions from the user
        segments: Optional new audio segments
        segment_audio_data: Optional audio data for new segments

    Returns:
        (updated project, summary text)
    """
    logger.info("=" * 60)
    logger.info("AGENT: Starting refinement (director mode)")
    logger.info(f"AGENT: Instructions: {instructions}")
    logger.info(f"AGENT: Existing layers: {len(project.layers)}")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set! Add it to your .env file.")

    segment_audio_data = segment_audio_data or {}
    segment_data: dict[str, dict] = {}
    if segments:
        segment_data = await _pre_analyze(segments, segment_audio_data)

    # Run director in refinement mode
    from app.services.director import Director

    director = Director()
    summary = await director.run_refinement(
        project, instructions,
        segments, segment_data, segment_audio_data,
    )

    project.updated_at = datetime.now()

    logger.info(f"AGENT: Refinement summary: {summary}")
    logger.info(f"AGENT: Project now has {len(project.layers)} layers")
    logger.info("=" * 60)

    return project, summary
