from pathlib import Path
from fastapi import UploadFile


# Supported audio MIME types
SUPPORTED_AUDIO_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
}

# Extension to content type mapping
EXTENSION_TO_CONTENT_TYPE = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}


def get_content_type(filename: str) -> str:
    """
    Get the content type for an audio file based on its extension.

    Args:
        filename: Name of the audio file

    Returns:
        Content type string (defaults to audio/wav if unknown)
    """
    ext = Path(filename).suffix.lower()
    return EXTENSION_TO_CONTENT_TYPE.get(ext, "audio/wav")


async def validate_audio_file(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is a supported audio format.

    Args:
        file: FastAPI UploadFile

    Returns:
        True if valid audio file, False otherwise
    """
    # Check content type if provided
    if file.content_type:
        if file.content_type in SUPPORTED_AUDIO_TYPES:
            return True

    # Check file extension
    if file.filename:
        content_type = get_content_type(file.filename)
        if content_type in SUPPORTED_AUDIO_TYPES:
            return True

    return False


async def read_upload_file(file: UploadFile) -> bytes:
    """
    Read bytes from an UploadFile.

    Args:
        file: FastAPI UploadFile

    Returns:
        File contents as bytes
    """
    content = await file.read()
    await file.seek(0)  # Reset for potential re-reading
    return content
