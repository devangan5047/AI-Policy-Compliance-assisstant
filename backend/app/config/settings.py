from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    app_name: str = "AI Policy Compliance Assistant"
    api_prefix: str = "/api/v1"
    cors_origins: str = "*"

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openai/gpt-3.5-mini", alias="OPENROUTER_MODEL")
    openrouter_base_url: str | None = Field(default=None, alias="OPENROUTER_BASE_URL")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="policy_chunks", alias="QDRANT_COLLECTION")

    elasticsearch_url: str | None = Field(default=None, alias="ELASTICSEARCH_URL")
    elasticsearch_index: str = Field(default="policy_chunks", alias="ELASTICSEARCH_INDEX")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=160, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
