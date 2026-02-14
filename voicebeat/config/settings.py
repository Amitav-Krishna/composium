from pathlib import Path

from pydantic_settings import BaseSettings

# Get the voicebeat root directory (parent of config/)
VOICEBEAT_ROOT = Path(__file__).parent.parent
ENV_FILE = VOICEBEAT_ROOT.parent / ".env"


class Settings(BaseSettings):
    smallest_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    subagent_model: str = "minimax/minimax-m2.5"

    pulse_model: str = "pulse"
    pulse_language: str = "en"

    tts_voice_id: str = "james"
    tts_sample_rate: int = 24000

    samples_dir: Path = Path("./samples")
    output_dir: Path = Path("./output")

    default_bpm: int = 120

    cloudflare_r2_token: str = ""
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_public_url: str = ""
    r2_bucket_name: str = ""

    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"


settings = Settings()

# Debug: Print what was loaded
if not settings.smallest_api_key:
    print(f"WARNING: SMALLEST_API_KEY not found. Looked in: {ENV_FILE}")
    print(f"  File exists: {ENV_FILE.exists()}")
