from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent

class EnvSettings(BaseSettings):
    GOOGLE_API_KEY: Optional[str] = None
    MONGO_URI: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
