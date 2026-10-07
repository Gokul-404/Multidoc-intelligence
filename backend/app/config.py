"""
SentinelRAG Configuration
Centralised, environment-driven settings using Pydantic BaseSettings.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "SentinelRAG"
    app_version: str = "1.0.0"
    environment: Literal["development", "production", "testing"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── API Security ─────────────────────────────────────────────────────────
    api_key: str = Field(default="sentinel-dev-key-change-in-production")
    secret_key: str = Field(default="change-me-in-production-32-chars!!")
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # ── LLM & Embedding Providers ─────────────────────────────────────────────
    llm_provider: Literal["groq", "gemini"] = "groq"
    embedding_provider: Literal["fastembed", "gemini"] = "fastembed"

    # ── Groq API ─────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_model: str = "openai/gpt-oss-120b"

    # ── Google Gemini ─────────────────────────────────────────────────────────
    google_api_key: str = Field(default="")
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    embedding_dimensions: int = 384  # 384 for bge-small-en-v1.5, 768 for gemini

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "sentinelrag_chunks"
    qdrant_parent_collection: str = "sentinelrag_parents"
    qdrant_timeout: int = 30

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_cache_ttl: int = 3600          # 1 hour default
    redis_conversation_ttl: int = 86400  # 24 hours

    # ── Ingestion ─────────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = [".pdf"]
    allowed_mime_types: list[str] = ["application/pdf"]

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 2048
    parent_chunk_overlap: int = 128
    min_chunk_length: int = 50

    # ── OCR ──────────────────────────────────────────────────────────────────
    ocr_engine: Literal["tesseract", "easyocr", "paddleocr"] = "tesseract"
    ocr_min_text_ratio: float = 0.05   # chars / page_area threshold
    tesseract_cmd: str = "tesseract"

    # ── Retrieval ────────────────────────────────────────────────────────────
    dense_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_top_k: int = 30
    rerank_top_k: int = 6
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_k: int = 60               # RRF constant

    # ── Reranking ─────────────────────────────────────────────────────────────
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"  # FlashRank

    # ── Generation ───────────────────────────────────────────────────────────
    max_context_tokens: int = 8192
    max_answer_tokens: int = 2048
    temperature: float = 0.0
    max_retries: int = 2

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_requests: int = 60
    rate_limit_window: int = 60  # seconds

    # ── Tenant ───────────────────────────────────────────────────────────────
    default_tenant: str = "default"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("google_api_key")
    @classmethod
    def validate_google_api_key(cls, v: str) -> str:
        if not v:
            import warnings
            warnings.warn(
                "GOOGLE_API_KEY is not set. AI features will not work.",
                stacklevel=2,
            )
        return v

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def upload_path(self):
        import pathlib
        p = pathlib.Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
