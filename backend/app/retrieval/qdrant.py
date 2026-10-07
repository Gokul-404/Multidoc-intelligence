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
        """Insert/update child chunks into the main collection."""
        client = await self._get_client()
        await self.ensure_collections()

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
            # Use deterministic ID based on chunk_id for idempotent upserts
            import hashlib
            point_id = int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16) % (2**63)
            points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        if points:
            await client.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
            )
            logger.debug("Upserted %d child chunks to Qdrant", len(points))

    async def upsert_parents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        tenant_id: str,
    ) -> None:
        """Insert/update parent chunks into the parent collection."""
        client = await self._get_client()
        await self.ensure_collections()

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
            import hashlib
            point_id = int(hashlib.md5(parent_id.encode()).hexdigest()[:16], 16) % (2**63)
            points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        if points:
            await client.upsert(
                collection_name=settings.qdrant_parent_collection,
                points=points,
            )
            logger.debug("Upserted %d parent chunks to Qdrant", len(points))

    async def search(
        self,
        query: str,
        tenant_id: str,
        document_ids: Optional[list[str]] = None,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Dense semantic search with mandatory tenant filter.
        Returns up to k results, always scoped to the requesting tenant.
        """
        client = await self._get_client()
        query_vector = self._embeddings.embed_query(query)
        query_filter = self._build_tenant_filter(tenant_id, document_ids)

        try:
            results = await client.search(
                collection_name=settings.qdrant_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=k,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            return []

        chunks = []
        for r in results:
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
            chunks.append(RetrievedChunk(document=doc, score=r.score, chunk_id=payload.get("chunk_id", "")))
        return chunks

    async def get_parent(
        self,
        parent_id: str,
        tenant_id: str,
    ) -> Optional[Document]:
        """Retrieve a parent chunk by ID, with tenant verification."""
        client = await self._get_client()
        query_filter = self._build_tenant_filter(tenant_id)
        query_filter.must.append(
            FieldCondition(key="parent_id", match=MatchValue(value=parent_id))
        )

        try:
            results = await client.scroll(
                collection_name=settings.qdrant_parent_collection,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
            )
            points = results[0]
            if not points:
                return None
            payload = points[0].payload or {}
            # Strict tenant check
            if payload.get("tenant_id") != tenant_id:
                logger.warning("Cross-tenant parent access attempt denied: %s", parent_id)
                return None
            return Document(
                page_content=payload.get("text", ""),
                metadata={**payload},
            )
        except Exception as exc:
            logger.error("Parent retrieval failed for %s: %s", parent_id, exc)
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
        logger.info("Deleted document %s from Qdrant", document_id)

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
