"""LangChain Embedding Service – wraps Google text-embedding-004."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Centralised embedding abstraction using LangChain interface."""

    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        self._model_name = model or settings.gemini_embedding_model
        self._dimensions = settings.embedding_dimensions
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._model_name,
            google_api_key=settings.google_api_key,
        )
        logger.info("EmbeddingService initialised with model=%s", self._model_name)

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

    # Expose the raw LangChain embeddings object for integrations that need it
    @property
    def langchain_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return self._embeddings


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
