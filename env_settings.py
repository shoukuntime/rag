from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class EnvSettings(BaseSettings):
    GOOGLE_API_KEY: Optional[str] = None
    MODEL_NAME: Optional[str] = "models/gemini-2.5-flash"
    EMBEDDING_MODEL: Optional[str] = "models/gemini-embedding-001"

    MONGO_URI: Optional[str] = None
    DB_NAME: Optional[str] = "rag_db"
    COLLECTION_NAME: Optional[str] = "rag_collection"
    INDEX_NAME: Optional[str] = "vector_index"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
