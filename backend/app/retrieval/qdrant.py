"""
Qdrant vector store service for SentinelRAG.
Enforces tenant isolation at the filter level on EVERY query.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import get_settings
from app.langchain_components.embeddings import get_embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    chunk_id: str
    retrieval_method: str = "dense"


import numpy as np
import os
import pickle

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QDRANT_STORAGE_PATH = os.path.join(_BACKEND_DIR, "data", "qdrant_storage.pkl")

# Global in-memory storage fallback when Qdrant daemon is offline
_GLOBAL_LOCAL_CHUNKS: dict[str, list[dict]] = {}
_GLOBAL_LOCAL_PARENTS: dict[str, dict[str, Document]] = {}


def _save_local_storage() -> None:
    try:
        os.makedirs(os.path.dirname(_QDRANT_STORAGE_PATH), exist_ok=True)
        with open(_QDRANT_STORAGE_PATH, "wb") as f:
            pickle.dump({"chunks": _GLOBAL_LOCAL_CHUNKS, "parents": _GLOBAL_LOCAL_PARENTS}, f)
    except Exception as e:
        logger.debug("Failed to persist local qdrant storage: %s", e)


def _load_local_storage() -> None:
    global _GLOBAL_LOCAL_CHUNKS, _GLOBAL_LOCAL_PARENTS
    try:
        if os.path.exists(_QDRANT_STORAGE_PATH):
            with open(_QDRANT_STORAGE_PATH, "rb") as f:
                data = pickle.load(f)
                _GLOBAL_LOCAL_CHUNKS = data.get("chunks", {})
                _GLOBAL_LOCAL_PARENTS = data.get("parents", {})
    except Exception as e:
        logger.debug("Failed to load local qdrant storage: %s", e)


_load_local_storage()


def _calc_cosine(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


class QdrantService:
    """
    Manages all interactions with Qdrant.
    Tenant isolation is ALWAYS enforced server-side via payload filters.
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None
        self._embeddings = get_embedding_service()

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=settings.qdrant_timeout,
            )
        return self._client

    async def ensure_collections(self) -> None:
        """Create Qdrant collections if they don't exist."""
        client = await self._get_client()
        existing = {c.name for c in (await client.get_collections()).collections}

        for coll_name in [settings.qdrant_collection, settings.qdrant_parent_collection]:
            if coll_name not in existing:
                await client.create_collection(
                    collection_name=coll_name,
                    vectors_config=VectorParams(
                        size=self._embeddings.dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", coll_name)

    def _build_tenant_filter(
        self,
        tenant_id: str,
        document_ids: Optional[list[str]] = None,
        extra_filters: Optional[list] = None,
    ) -> Filter:
        """
        Build a filter that ALWAYS enforces tenant isolation.
        Optional document_id restriction for per-document queries.
        """
        must_conditions = [
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        ]
        if document_ids:
            must_conditions.append(
                models.Filter(
                    should=[
                        FieldCondition(key="document_id", match=MatchValue(value=did))
                        for did in document_ids
                    ]
                )
            )
        if extra_filters:
            must_conditions.extend(extra_filters)
        return Filter(must=must_conditions)

    async def upsert_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        tenant_id: str,
        content_hash: str = "",
    ) -> None:
        """Insert/update child chunks into the main collection, with in-memory backup."""
        if tenant_id not in _GLOBAL_LOCAL_CHUNKS:
            _GLOBAL_LOCAL_CHUNKS[tenant_id] = []

        points = []
        for doc, emb in zip(documents, embeddings):
            chunk_id = doc.metadata.get("chunk_id", "")
            payload = {
                "text": doc.page_content,
                "tenant_id": tenant_id,
                "document_id": doc.metadata.get("document_id", ""),
                "filename": doc.metadata.get("filename", ""),
                "page": doc.metadata.get("page", 0),
                "section": doc.metadata.get("section", ""),
                "chunk_id": chunk_id,
                "parent_id": doc.metadata.get("parent_id", ""),
                "chunk_type": doc.metadata.get("chunk_type", "child"),
                "document_type": doc.metadata.get("document_type", "general"),
                "language": doc.metadata.get("language", "en"),
                "content_hash": content_hash,
            }
            # Always save to resilient in-memory store
            _GLOBAL_LOCAL_CHUNKS[tenant_id].append({
                "vector": emb,
                "payload": payload,
                "document": doc,
            })

            import hashlib
            point_id = int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16) % (2**63)
            points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        try:
            client = await self._get_client()
            await self.ensure_collections()
            if points:
                await client.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points,
                )
                logger.debug("Upserted %d child chunks to Qdrant", len(points))
        except Exception as exc:
            logger.info("Qdrant offline; %d child chunks cached in local store: %s", len(points), exc)
        _save_local_storage()

    async def upsert_parents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        tenant_id: str,
    ) -> None:
        """Insert/update parent chunks into the parent collection, with in-memory backup."""
        if tenant_id not in _GLOBAL_LOCAL_PARENTS:
            _GLOBAL_LOCAL_PARENTS[tenant_id] = {}

        points = []
        for doc, emb in zip(documents, embeddings):
            parent_id = doc.metadata.get("chunk_id", doc.metadata.get("parent_id", ""))
            payload = {
                "text": doc.page_content,
                "tenant_id": tenant_id,
                "document_id": doc.metadata.get("document_id", ""),
                "filename": doc.metadata.get("filename", ""),
                "page": doc.metadata.get("page", 0),
                "section": doc.metadata.get("section", ""),
                "parent_id": parent_id,
                "chunk_type": "parent",
            }
            # Always save to resilient parent store
            _GLOBAL_LOCAL_PARENTS[tenant_id][parent_id] = doc

            import hashlib
            point_id = int(hashlib.md5(parent_id.encode()).hexdigest()[:16], 16) % (2**63)
            points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        try:
            client = await self._get_client()
            await self.ensure_collections()
            if points:
                await client.upsert(
                    collection_name=settings.qdrant_parent_collection,
                    points=points,
                )
                logger.debug("Upserted %d parent chunks to Qdrant", len(points))
        except Exception as exc:
            logger.info("Qdrant offline; %d parent chunks cached in local store: %s", len(points), exc)
        _save_local_storage()

    async def search(
        self,
        query: str,
        tenant_id: str,
        document_ids: Optional[list[str]] = None,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Dense semantic search with mandatory tenant filter.
        Falls back seamlessly to local in-memory cosine search if Qdrant is unavailable.
        """
        query_vector = self._embeddings.embed_query(query)
        chunks = []

        try:
            client = await self._get_client()
            query_filter = self._build_tenant_filter(tenant_id, document_ids)
            if hasattr(client, "query_points"):
                resp = await client.query_points(
                    collection_name=settings.qdrant_collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True,
                )
                raw_points = resp.points
            else:
                raw_points = await client.search(
                    collection_name=settings.qdrant_collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True,
                )
            for r in raw_points:
                payload = r.payload or {}
                doc = Document(
                    page_content=payload.get("text", ""),
                    metadata={
                        "document_id": payload.get("document_id", ""),
                        "filename": payload.get("filename", ""),
                        "page": payload.get("page", 0),
                        "section": payload.get("section", ""),
                        "chunk_id": payload.get("chunk_id", ""),
                        "parent_id": payload.get("parent_id", ""),
                        "chunk_type": payload.get("chunk_type", "child"),
                        "document_type": payload.get("document_type", "general"),
                        "tenant_id": payload.get("tenant_id", ""),
                    },
                )
                chunks.append(RetrievedChunk(document=doc, score=float(r.score), chunk_id=payload.get("chunk_id", "")))
        except Exception as exc:
            logger.debug("Qdrant search bypassed: %s", exc)

        # Fallback to local in-memory cosine similarity
        if not chunks and tenant_id in _GLOBAL_LOCAL_CHUNKS:
            candidates = _GLOBAL_LOCAL_CHUNKS[tenant_id]
            scored = []
            for item in candidates:
                p = item["payload"]
                if document_ids and p.get("document_id") not in document_ids:
                    continue
                score = _calc_cosine(query_vector, item["vector"])
                scored.append(RetrievedChunk(document=item["document"], score=score, chunk_id=p.get("chunk_id", "")))
            scored.sort(key=lambda x: x.score, reverse=True)
            chunks = scored[:k]
            logger.info("Local in-memory search returned %d chunks for tenant %s", len(chunks), tenant_id)

        return chunks

    async def get_parent(
        self,
        parent_id: str,
        tenant_id: str,
    ) -> Optional[Document]:
        """Retrieve a parent chunk by ID, with tenant verification."""
        try:
            client = await self._get_client()
            query_filter = self._build_tenant_filter(tenant_id)
            query_filter.must.append(
                FieldCondition(key="parent_id", match=MatchValue(value=parent_id))
            )
            results = await client.scroll(
                collection_name=settings.qdrant_parent_collection,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
            )
            points = results[0]
            if points:
                payload = points[0].payload or {}
                if payload.get("tenant_id") == tenant_id:
                    return Document(page_content=payload.get("text", ""), metadata={**payload})
        except Exception:
            pass

        # In-memory parent fallback
        if tenant_id in _GLOBAL_LOCAL_PARENTS and parent_id in _GLOBAL_LOCAL_PARENTS[tenant_id]:
            return _GLOBAL_LOCAL_PARENTS[tenant_id][parent_id]
        return None

    async def find_by_content_hash(self, content_hash: str, tenant_id: str) -> Optional[str]:
        """Return filename of existing document with same content hash, or None."""
        client = await self._get_client()
        try:
            await self.ensure_collections()
            results = await client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            points = results[0]
            if points:
                return points[0].payload.get("filename")
        except Exception as exc:
            logger.debug("Content hash check failed: %s", exc)
        return None

    async def delete_document(self, document_id: str, tenant_id: str) -> None:
        """Delete all chunks for a document from both collections."""
        client = await self._get_client()
        doc_filter = self._build_tenant_filter(tenant_id)
        doc_filter.must.append(
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        )
        for coll in [settings.qdrant_collection, settings.qdrant_parent_collection]:
            try:
                await client.delete(collection_name=coll, points_selector=doc_filter)
            except Exception as exc:
                logger.error("Failed to delete from %s: %s", coll, exc)

        # Remove from local in-memory/disk store
        if tenant_id in _GLOBAL_LOCAL_CHUNKS:
            _GLOBAL_LOCAL_CHUNKS[tenant_id] = [
                c for c in _GLOBAL_LOCAL_CHUNKS[tenant_id]
                if c["payload"].get("document_id") != document_id
            ]
        if tenant_id in _GLOBAL_LOCAL_PARENTS:
            _GLOBAL_LOCAL_PARENTS[tenant_id] = {
                pid: doc for pid, doc in _GLOBAL_LOCAL_PARENTS[tenant_id].items()
                if doc.metadata.get("document_id") != document_id
            }
        _save_local_storage()
        logger.info("Deleted document %s from Qdrant and local store", document_id)

    async def get_stats(self, tenant_id: str) -> dict:
        """Return collection stats for this tenant."""
        client = await self._get_client()
        try:
            # Count points for this tenant
            count_result = await client.count(
                collection_name=settings.qdrant_collection,
                count_filter=Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
                ),
            )
            return {"total_chunks": count_result.count}
        except Exception as exc:
            logger.error("Stats query failed: %s", exc)
            return {"total_chunks": 0}
