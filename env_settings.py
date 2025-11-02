from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class EnvSettings(BaseSettings):
    GOOGLE_API_KEY: Optional[str] = None
    MODEL_NAME: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None

    MONGO_URI: Optional[str] = None
    DB_NAME: Optional[str] = None
    COLLECTION_NAME: Optional[str] = None

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
