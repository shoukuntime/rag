from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class EnvSettings(BaseSettings):
    GOOGLE_API_KEY: str
    MODEL_NAME: str
    EMBEDDING_MODEL: str

    POSTGRES_URI: str
    COLLECTION_NAME: str

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
