"""Application configuration loaded from environment variables or .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    searxng_base_url: str
    searxng_timeout_seconds: float = Field(default=10, gt=0)
    searxng_max_retries: int = Field(default=2, ge=0)
    searxng_retry_backoff_seconds: float = Field(default=0.25, ge=0)
    searxng_max_results: int = Field(default=10, gt=0)
    document_timeout_seconds: float = Field(default=20, gt=0)
    document_store_dir: Path = Path(".deepresearch/documents")
    retrieval_chunk_size_chars: int = Field(default=1200, gt=0)
    retrieval_chunk_overlap_chars: int = Field(default=200, ge=0)
    retrieval_bm25_top_k: int = Field(default=8, gt=0)
    retrieval_top_n: int = Field(default=3, gt=0)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("searxng_base_url")
    @classmethod
    def validate_searxng_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("SEARXNG_BASE_URL must use http:// or https://.")
        return normalized
