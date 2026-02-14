from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    smallest_api_key: str = ""
    anthropic_api_key: str = ""

    pulse_model: str = "pulse"
    pulse_language: str = "en"

    tts_voice_id: str = "jasmine"
    tts_sample_rate: int = 24000

    samples_dir: Path = Path("./samples")
    output_dir: Path = Path("./output")

    default_bpm: int = 120

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
