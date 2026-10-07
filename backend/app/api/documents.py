"""
Document upload and management API endpoints.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status

from app.cache.query_cache import QueryCache
from app.cache.redis_client import get_redis_client
from app.config import get_settings
from app.ingestion.loaders import IngestionPipeline
from app.models.schemas import (
    ChunkResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentResponse,
    DocumentStatus,
)
from app.security.auth import require_api_key, sanitize_filename

router = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()
logger = logging.getLogger(__name__)

# Simple in-memory tenant extraction (replace with JWT in production)
def get_tenant_id(request: Request) -> str:
    return request.headers.get("X-Tenant-ID", settings.default_tenant)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: str = Depends(require_api_key),
):
    """
    Upload a PDF document for processing.
    Processing happens in the background.
    Poll GET /documents/{id} for status.
    """
    tenant_id = get_tenant_id(request)
    safe_filename = sanitize_filename(file.filename or "document.pdf")

    # Read file bytes
    file_bytes = await file.read()

    # Start ingestion as background task
    pipeline = IngestionPipeline()

    async def _ingest():
        try:
            await pipeline.ingest(
                file_bytes=file_bytes,
                filename=safe_filename,
                tenant_id=tenant_id,
            )
        except ValueError as exc:
            logger.warning("Ingestion rejected: %s", exc)
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)

    background_tasks.add_task(_ingest)

    return {
        "message": "Upload accepted. Processing started.",
        "filename": safe_filename,
        "tenant_id": tenant_id,
    }


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    search: Optional[str] = None,
    doc_type: Optional[str] = None,
    status_filter: Optional[str] = None,
    _: str = Depends(require_api_key),
):
    """List all documents for the tenant."""
    tenant_id = get_tenant_id(request)
    redis = get_redis_client()

    try:
        doc_ids = await redis.smembers(f"tenant_docs:{tenant_id}")
    except Exception:
        doc_ids = set()

    documents = []
    for doc_id in doc_ids:
        try:
            raw = await redis.get(f"doc_meta:{tenant_id}:{doc_id}")
            if raw:
                meta = DocumentMetadata.model_validate_json(raw)
                # Apply filters
                if search and search.lower() not in meta.filename.lower():
                    continue
                if doc_type and meta.document_type.value != doc_type:
                    continue
                if status_filter and meta.status.value != status_filter:
                    continue
                documents.append(
                    DocumentResponse(
                        document_id=meta.document_id,
                        filename=meta.filename,
                        status=meta.status,
                        document_type=meta.document_type,
                        pages=meta.pages,
                        chunks=meta.chunks,
                        ocr_used=meta.ocr_used,
                        uploaded_at=meta.uploaded_at,
                        processing_completed_at=meta.processing_completed_at,
                        error_message=meta.error_message,
                    )
                )
        except Exception as exc:
            logger.warning("Could not load doc %s: %s", doc_id, exc)

    documents.sort(key=lambda d: d.uploaded_at, reverse=True)
    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    request: Request,
    _: str = Depends(require_api_key),
):
    """Get document metadata and processing status."""
    tenant_id = get_tenant_id(request)
    redis = get_redis_client()

    # Check Redis for full metadata
    raw = await redis.get(f"doc_meta:{tenant_id}:{document_id}")
    if raw:
        return DocumentMetadata.model_validate_json(raw)

    # Fall back to status-only from ingestion progress
    status_data = await redis.hgetall(f"doc_status:{document_id}")
    if status_data and status_data.get("tenant_id") == tenant_id:
        return DocumentMetadata(
            document_id=document_id,
            filename=status_data.get("filename", "unknown"),
            tenant_id=tenant_id,
            status=DocumentStatus(status_data.get("status", "uploading")),
            error_message=status_data.get("error") or None,
        )

    raise HTTPException(status_code=404, detail="Document not found")


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    request: Request,
    _: str = Depends(require_api_key),
):
    """Delete a document and all associated data."""
    tenant_id = get_tenant_id(request)

    # Verify ownership
    redis = get_redis_client()
    is_member = await redis.sismember(f"tenant_docs:{tenant_id}", document_id)
    if not is_member:
        raise HTTPException(status_code=404, detail="Document not found")

    pipeline = IngestionPipeline()
    await pipeline.delete_document(document_id, tenant_id)

    # Invalidate query cache
    cache = QueryCache()
    await cache.invalidate_for_document(document_id, tenant_id)


@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(
    document_id: str,
    request: Request,
    page: Optional[int] = None,
    _: str = Depends(require_api_key),
):
    """Get chunks for a document. Optionally filter by page number."""
    tenant_id = get_tenant_id(request)

    # Verify ownership
    redis = get_redis_client()
    is_member = await redis.sismember(f"tenant_docs:{tenant_id}", document_id)
    if not is_member:
        raise HTTPException(status_code=404, detail="Document not found")

    from app.retrieval.qdrant import QdrantService
    qdrant = QdrantService()
    results = await qdrant.search(
        query="",  # Empty query for listing
        tenant_id=tenant_id,
        document_ids=[document_id],
        k=200,
    )

    chunks = []
    for r in results:
        doc = r.document
        if page is not None and doc.metadata.get("page") != page:
            continue
        chunks.append(
            ChunkResponse(
                chunk_id=doc.metadata.get("chunk_id", ""),
                document_id=doc.metadata.get("document_id", ""),
                filename=doc.metadata.get("filename", ""),
                page=doc.metadata.get("page", 0),
                section=doc.metadata.get("section"),
                content=doc.page_content[:500],
                chunk_type=doc.metadata.get("chunk_type", "child"),
            )
        )

    return chunks


@router.get("/{document_id}/status")
async def get_processing_status(
    document_id: str,
    request: Request,
    _: str = Depends(require_api_key),
):
    """Poll processing status for a document."""
    tenant_id = get_tenant_id(request)
    redis = get_redis_client()
    status_data = await redis.hgetall(f"doc_status:{document_id}")

    if not status_data:
        raise HTTPException(status_code=404, detail="Document not found or status expired")

    return status_data
