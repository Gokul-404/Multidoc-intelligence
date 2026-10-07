"""
LangChain-compatible retriever interfaces for SentinelRAG.
Wraps dense and BM25 retrievers behind LangChain's BaseRetriever interface.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QdrantRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by Qdrant dense search."""

    qdrant_store: Any  # QdrantVectorStore instance
    tenant_id: str
    document_ids: Optional[list[str]] = None
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        import asyncio
        from app.retrieval.qdrant import QdrantService
        # Delegate to the service layer to honour tenant filtering
        service = QdrantService()
        results = asyncio.get_event_loop().run_until_complete(
            service.search(
                query=query,
                tenant_id=self.tenant_id,
                document_ids=self.document_ids,
                k=self.k,
            )
        )
        return [r.document for r in results]


class BM25Retriever(BaseRetriever):
    """LangChain-compatible BM25 retriever."""

    bm25_service: Any  # BM25Service instance
    tenant_id: str
    document_ids: Optional[list[str]] = None
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        import asyncio
        results = asyncio.get_event_loop().run_until_complete(
            self.bm25_service.search(
                query=query,
                tenant_id=self.tenant_id,
                document_ids=self.document_ids,
                k=self.k,
            )
        )
        return [r.document for r in results]
