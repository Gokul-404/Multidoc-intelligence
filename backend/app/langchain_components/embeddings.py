"""LangChain Embedding Service – wraps Google text-embedding-004."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional, Any

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # type: ignore[assignment, misc]

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


class FastEmbedAdapter:
    """LangChain-compatible wrapper around FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        if TextEmbedding is None:
            raise ImportError("fastembed is not installed.")
        self._model = TextEmbedding(model_name=model_name)
        self._dimensions = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(vec) for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(next(self._model.embed([text])))


class EmbeddingService:
    """Centralised embedding abstraction supporting FastEmbed (local, free) and Gemini."""

    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        self._provider = settings.embedding_provider

        if self._provider == "fastembed" and TextEmbedding is not None:
            self._model_name = model or "BAAI/bge-small-en-v1.5"
            self._dimensions = 384
            self._embeddings = FastEmbedAdapter(model_name=self._model_name)
            logger.info("EmbeddingService initialised with FastEmbed model=%s (dim=%d)", self._model_name, self._dimensions)
        else:
            self._model_name = model or settings.gemini_embedding_model
            self._dimensions = settings.embedding_dimensions
            api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "placeholder-key"
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self._model_name,
                google_api_key=api_key,
            )
            logger.info("EmbeddingService initialised with Gemini model=%s (dim=%d)", self._model_name, self._dimensions)

    # LangChain-compatible interface -----------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple documents."""
        if not texts:
            return []
        logger.debug("Embedding %d documents", len(texts))
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        logger.debug("Embedding query: %.80s", text)
        return self._embeddings.embed_query(text)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langchain_embeddings(self) -> Any:
        return self._embeddings


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
