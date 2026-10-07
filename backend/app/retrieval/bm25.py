"""
BM25 sparse retrieval for SentinelRAG.
In-memory per-tenant index backed by Redis for persistence.
Returns LangChain Document objects.
"""
from __future__ import annotations

import json
import logging
import pickle
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.cache.redis_client import get_redis_client
from app.config import get_settings
from app.retrieval.qdrant import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

_BM25_INDEX_PREFIX = "bm25_index"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


_GLOBAL_BM25_CACHE: dict[str, tuple[BM25Okapi, list[Document]]] = {}


class BM25Service:
    """
    BM25 retrieval service.
    - Indexes are stored in memory and persisted to Redis.
    - Documents (page_content + metadata) are stored alongside the index.
    - Every query enforces tenant isolation.
    """

    def __init__(self) -> None:
        self._redis = get_redis_client()

    def _index_key(self, tenant_id: str) -> str:
        return f"{_BM25_INDEX_PREFIX}:{tenant_id}"

    def _docs_key(self, tenant_id: str) -> str:
        return f"{_BM25_INDEX_PREFIX}:docs:{tenant_id}"

    async def _load_index(self, tenant_id: str) -> Optional[tuple[BM25Okapi, list[Document]]]:
        """Load BM25 index and documents from cache or Redis."""
        if tenant_id in _GLOBAL_BM25_CACHE:
            return _GLOBAL_BM25_CACHE[tenant_id]
        try:
            index_bytes = await self._redis.get(self._index_key(tenant_id))
            docs_bytes = await self._redis.get(self._docs_key(tenant_id))
            if index_bytes and docs_bytes:
                index = pickle.loads(index_bytes)
                docs = pickle.loads(docs_bytes)
                _GLOBAL_BM25_CACHE[tenant_id] = (index, docs)
                return index, docs
        except Exception as exc:
            logger.debug("BM25 cache read: %s", exc)
        return _GLOBAL_BM25_CACHE.get(tenant_id)

    async def _save_index(
        self, tenant_id: str, index: BM25Okapi, docs: list[Document]
    ) -> None:
        """Persist BM25 index to cache and Redis."""
        _GLOBAL_BM25_CACHE[tenant_id] = (index, docs)
        try:
            await self._redis.set(
                self._index_key(tenant_id),
                pickle.dumps(index),
                ex=86400 * 7,
            )
            await self._redis.set(
                self._docs_key(tenant_id),
                pickle.dumps(docs),
                ex=86400 * 7,
            )
        except Exception as exc:
            logger.debug("Redis BM25 write bypassed: %s", exc)

    async def add_documents(self, documents: list[Document], tenant_id: str) -> None:
        """
        Add documents to the BM25 index for a tenant.
        Only includes documents belonging to this tenant (validated from metadata).
        Rebuilds the full index each time (acceptable for document counts up to ~50k).
        """
        # Filter to only this tenant's documents
        tenant_docs = [
            d for d in documents
            if d.metadata.get("tenant_id", tenant_id) == tenant_id
        ]
        if not tenant_docs:
            return

        # Load existing docs and append
        existing = await self._load_index(tenant_id)
        existing_docs: list[Document] = existing[1] if existing else []

        # Merge: replace if same chunk_id exists
        existing_ids = {d.metadata.get("chunk_id", ""): i for i, d in enumerate(existing_docs)}
        for new_doc in tenant_docs:
            cid = new_doc.metadata.get("chunk_id", "")
            if cid in existing_ids:
                existing_docs[existing_ids[cid]] = new_doc
            else:
                existing_docs.append(new_doc)

        # Rebuild index
        corpus = [_tokenize(d.page_content) for d in existing_docs]
        index = BM25Okapi(corpus)
        await self._save_index(tenant_id, index, existing_docs)
        logger.info("BM25 index updated for tenant %s: %d documents", tenant_id, len(existing_docs))

    async def search(
        self,
        query: str,
        tenant_id: str,
        document_ids: Optional[list[str]] = None,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        BM25 keyword search with tenant isolation.
        Optionally restrict to specific document_ids.
        """
        loaded = await self._load_index(tenant_id)
        if loaded is None:
            logger.debug("No BM25 index for tenant %s", tenant_id)
            return []

        index, docs = loaded
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = index.get_scores(tokenized_query)

        # Create scored pairs and filter by tenant + optional document_ids
        ranked: list[tuple[float, int]] = []
        for i, score in enumerate(scores):
            doc = docs[i]
            # Tenant isolation enforcement
            if doc.metadata.get("tenant_id", tenant_id) != tenant_id:
                continue
            # Optional document filter
            if document_ids and doc.metadata.get("document_id") not in document_ids:
                continue
            if score > 0:
                ranked.append((score, i))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top_k = ranked[:k]

        results = []
        for score, idx in top_k:
            doc = docs[idx]
            results.append(
                RetrievedChunk(
                    document=doc,
                    score=float(score),
                    chunk_id=doc.metadata.get("chunk_id", ""),
                )
            )
        return results

    async def remove_document(self, document_id: str, tenant_id: str) -> None:
        """Remove all chunks belonging to a document from the BM25 index."""
        loaded = await self._load_index(tenant_id)
        if loaded is None:
            return

        _, docs = loaded
        remaining = [d for d in docs if d.metadata.get("document_id") != document_id]

        if not remaining:
            await self._redis.delete(self._index_key(tenant_id))
            await self._redis.delete(self._docs_key(tenant_id))
            _GLOBAL_BM25_CACHE.pop(tenant_id, None)
            return

        corpus = [_tokenize(d.page_content) for d in remaining]
        new_index = BM25Okapi(corpus)
        await self._save_index(tenant_id, new_index, remaining)
        logger.info("BM25: removed document %s for tenant %s", document_id, tenant_id)
