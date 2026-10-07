"""
Document ingestion orchestrator for SentinelRAG.
Coordinates: validation → OCR → layout → chunking → embedding → indexing.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.cache.redis_client import get_redis_client
from app.config import get_settings
from app.ingestion.chunking import chunk_document
from app.ingestion.layout import extract_layout
from app.ingestion.metadata import (
    build_chunk_metadata,
    detect_document_type,
    detect_language,
    extract_pdf_metadata,
)
from app.ingestion.ocr import run_ocr
from app.ingestion.pdf_parser import validate_pdf
from app.langchain_components.embeddings import get_embedding_service
from app.models.schemas import DocumentMetadata, DocumentStatus, DocumentType
from app.retrieval.bm25 import BM25Service
from app.retrieval.qdrant import QdrantService

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline for a PDF document.
    Reports progress to Redis so the frontend can poll status.
    """

    def __init__(self) -> None:
        self._qdrant = QdrantService()
        self._bm25 = BM25Service()
        self._embeddings = get_embedding_service()
        self._redis = get_redis_client()

    async def _set_status(
        self,
        document_id: str,
        status: DocumentStatus,
        metadata: dict,
        error: Optional[str] = None,
    ) -> None:
        """Write ingestion status to Redis."""
        payload = {
            "status": status.value,
            "document_id": document_id,
            "error": error or "",
            **{k: str(v) if v is not None else "" for k, v in metadata.items()},
        }
        try:
            await self._redis.hset(f"doc_status:{document_id}", mapping=payload)
            await self._redis.expire(f"doc_status:{document_id}", 86400)
        except Exception as exc:
            logger.warning("Could not write status to Redis: %s", exc)

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        tenant_id: str,
        document_id: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Full ingestion pipeline.
        Returns completed DocumentMetadata or raises on unrecoverable error.
        """
        doc_id = document_id or str(uuid.uuid4())
        start_time = time.monotonic()
        meta: dict = {
            "filename": filename,
            "tenant_id": tenant_id,
            "document_id": doc_id,
            "pages": 0,
            "chunks": 0,
            "ocr_used": False,
        }

        # ── 1. Validation ─────────────────────────────────────────────────────
        await self._set_status(doc_id, DocumentStatus.UPLOADING, meta)
        validation = validate_pdf(file_bytes, filename)
        if not validation.valid:
            await self._set_status(doc_id, DocumentStatus.FAILED, meta, error=validation.error)
            raise ValueError(validation.error)

        # Duplicate detection
        dup = await self._qdrant.find_by_content_hash(validation.content_hash, tenant_id)
        if dup:
            raise ValueError(f"Document already indexed as '{dup}'.")

        meta["pages"] = validation.pages

        # ── 2. Text extraction / OCR ──────────────────────────────────────────
        await self._set_status(doc_id, DocumentStatus.EXTRACTING, meta)
        ocr_result = run_ocr(file_bytes)
        if not ocr_result.pages:
            await self._set_status(doc_id, DocumentStatus.FAILED, meta, error="Could not extract any text from PDF.")
            raise ValueError("Text extraction produced no content.")

        page_texts = {p.page_number: p.text for p in ocr_result.pages if p.text.strip()}
        meta["ocr_used"] = ocr_result.ocr_used
        if ocr_result.low_quality:
            logger.warning("Low OCR quality for document %s. Results may be poor.", doc_id)

        # ── 3. Layout detection ───────────────────────────────────────────────
        await self._set_status(doc_id, DocumentStatus.DETECTING_STRUCTURE, meta)
        layout = extract_layout(page_texts)

        # Full text sample for metadata detection
        full_text = " ".join(p.text for p in ocr_result.pages)[:5000]

        # Metadata enrichment
        pdf_meta = extract_pdf_metadata(file_bytes)
        doc_type = detect_document_type(full_text)
        language = detect_language(full_text)

        # ── 4. Chunking ───────────────────────────────────────────────────────
        await self._set_status(doc_id, DocumentStatus.CHUNKING, meta)
        chunking_result = chunk_document(
            layout=layout,
            document_id=doc_id,
            filename=filename,
            tenant_id=tenant_id,
            document_type=doc_type,
            language=language,
            author=pdf_meta.get("author"),
            creation_date=pdf_meta.get("creation_date"),
        )

        if not chunking_result.child_docs:
            await self._set_status(doc_id, DocumentStatus.FAILED, meta, error="No usable chunks produced from document.")
            raise ValueError("Chunking produced no content.")

        meta["chunks"] = len(chunking_result.child_docs)

        # ── 5. Embeddings ─────────────────────────────────────────────────────
        await self._set_status(doc_id, DocumentStatus.EMBEDDING, meta)
        texts = [d.page_content for d in chunking_result.child_docs]
        embeddings = self._embeddings.embed_documents(texts)

        parent_texts = [d.page_content for d in chunking_result.parent_docs]
        parent_embeddings = self._embeddings.embed_documents(parent_texts)

        # ── 6. Indexing ───────────────────────────────────────────────────────
        await self._set_status(doc_id, DocumentStatus.INDEXING, meta)

        # Index child chunks in Qdrant
        await self._qdrant.upsert_documents(
            documents=chunking_result.child_docs,
            embeddings=embeddings,
            tenant_id=tenant_id,
            content_hash=validation.content_hash,
        )

        # Index parent chunks in Qdrant (separate collection)
        await self._qdrant.upsert_parents(
            documents=chunking_result.parent_docs,
            embeddings=parent_embeddings,
            tenant_id=tenant_id,
        )

        # Index all chunks in BM25
        all_chunks = chunking_result.child_docs + chunking_result.parent_docs
        await self._bm25.add_documents(all_chunks, tenant_id=tenant_id)

        # ── 7. Complete ───────────────────────────────────────────────────────
        elapsed_ms = (time.monotonic() - start_time) * 1000
        doc_metadata = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            tenant_id=tenant_id,
            status=DocumentStatus.READY,
            document_type=doc_type,
            pages=validation.pages,
            chunks=len(chunking_result.child_docs),
            parent_chunks=len(chunking_result.parent_docs),
            file_size_bytes=len(file_bytes),
            content_hash=validation.content_hash,
            ocr_used=ocr_result.ocr_used,
            ocr_confidence=ocr_result.avg_confidence,
            language=language,
            author=pdf_meta.get("author"),
            creation_date=pdf_meta.get("creation_date"),
            title=pdf_meta.get("title") or layout.title,
            processing_completed_at=datetime.utcnow(),
        )

        await self._set_status(doc_id, DocumentStatus.READY, {
            **meta,
            "pages": str(validation.pages),
            "chunks": str(len(chunking_result.child_docs)),
            "document_type": doc_type.value,
            "elapsed_ms": f"{elapsed_ms:.0f}",
        })

        # Store document metadata in Redis for fast retrieval
        await self._redis.set(
            f"doc_meta:{tenant_id}:{doc_id}",
            doc_metadata.model_dump_json(),
            ex=86400 * 30,
        )

        # Add to tenant document set
        await self._redis.sadd(f"tenant_docs:{tenant_id}", doc_id)

        logger.info(
            "Ingestion complete for %s: %d pages, %d chunks in %.0fms",
            filename, validation.pages, len(chunking_result.child_docs), elapsed_ms
        )
        return doc_metadata

    async def delete_document(self, document_id: str, tenant_id: str) -> None:
        """Remove document from all storage systems and invalidate cache."""
        # Qdrant
        await self._qdrant.delete_document(document_id, tenant_id)
        # BM25
        await self._bm25.remove_document(document_id, tenant_id)
        # Redis metadata
        await self._redis.delete(f"doc_meta:{tenant_id}:{document_id}")
        await self._redis.delete(f"doc_status:{document_id}")
        await self._redis.srem(f"tenant_docs:{tenant_id}", document_id)
        # Invalidate any cached query responses involving this document
        await self._redis.delete(f"cache:doc:{document_id}")
        logger.info("Deleted document %s from all stores", document_id)
