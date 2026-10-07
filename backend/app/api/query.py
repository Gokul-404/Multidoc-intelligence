"""
Query API endpoints for SentinelRAG.
Supports both regular and streaming responses.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.cache.query_cache import QueryCache
from app.cache.redis_client import get_redis_client
from app.config import get_settings
from app.graph.state import GraphState
from app.graph.workflow import get_workflow
from app.memory.conversation import ConversationManager
from app.models.schemas import (
    AuditResult,
    AuditStatus,
    QueryAnswer,
    QueryRequest,
    SystemStats,
)
from app.security.auth import check_rate_limit, require_api_key

router = APIRouter(prefix="/query", tags=["query"])
settings = get_settings()
logger = logging.getLogger(__name__)


def get_tenant_id(request: Request) -> str:
    return request.headers.get("X-Tenant-ID", settings.default_tenant)


async def _run_pipeline(
    question: str,
    tenant_id: str,
    document_ids: Optional[list[str]],
    conversation_id: Optional[str],
    enable_developer_mode: bool,
    retrieval_config,
) -> QueryAnswer:
    """Execute the LangGraph pipeline and return a QueryAnswer."""
    t0 = time.monotonic()

    # Load conversation memory
    conv_manager = ConversationManager()
    memory = None
    conversation_context = ""
    standalone_question = question

    if conversation_id:
        memory = await conv_manager.get(conversation_id, tenant_id)
        conversation_context = memory.get_context_string()
        standalone_question = memory.get_standalone_question(question)

    # Build initial graph state
    initial_state: GraphState = {
        "question": standalone_question,
        "original_question": question,
        "tenant_id": tenant_id,
        "document_ids": document_ids,
        "conversation_id": conversation_id,
        "conversation_context": conversation_context,
        "retrieval_config": retrieval_config,
        "enable_developer_mode": enable_developer_mode,
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "pipeline_stages": [],
        "rewritten_queries": [],
        "subqueries": [],
        "hybrid_results": [],
        "reranked_results": [],
        "parent_docs": [],
        "compressed_docs": [],
        "citations": [],
        "refused": False,
        "error": None,
    }

    # Run LangGraph workflow
    workflow = get_workflow()
    final_state: GraphState = workflow.invoke(initial_state)

    latency_ms = (time.monotonic() - t0) * 1000

    # Build response
    audit_result = final_state.get("audit_result") or AuditResult(
        status=AuditStatus.PASS if final_state.get("refused") else AuditStatus.PASS,
        claims_total=0,
        claims_supported=0,
        claims_unsupported=0,
        citations_valid=0,
        citations_invalid=0,
        has_numerical_claims=False,
        numerical_claims_verified=True,
        has_contradictions=False,
        confidence_score=0.0,
        grounded=bool(final_state.get("refused")),
    )

    answer_id = final_state.get("answer_id") or str(uuid.uuid4())

    query_answer = QueryAnswer(
        answer_id=answer_id,
        question=question,
        answer=final_state.get("answer", "I couldn't verify this from the uploaded documents."),
        citations=final_state.get("citations", []),
        audit=audit_result,
        query_analysis=final_state.get("query_analysis"),
        rewritten_queries=final_state.get("rewritten_queries", []),
        subqueries=final_state.get("subqueries", []),
        pipeline_stages=final_state.get("pipeline_stages", []) if enable_developer_mode else [],
        retry_count=final_state.get("retry_count", 0),
        latency_ms=latency_ms,
        cache_hit=False,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
    )

    # Update conversation memory
    if conversation_id:
        await conv_manager.add_turn(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_message=question,
            assistant_message=query_answer.answer,
        )

    # Store answer in Redis for evidence lookup
    redis = get_redis_client()
    await redis.setex(
        f"answer:{tenant_id}:{answer_id}",
        3600,
        query_answer.model_dump_json(),
    )

    return query_answer


@router.post("", response_model=QueryAnswer)
async def query(
    request: Request,
    body: QueryRequest,
    _: str = Depends(require_api_key),
):
    """Execute a query against uploaded documents."""
    tenant_id = get_tenant_id(request)

    # Rate limiting
    api_key = request.headers.get("X-API-Key", "")
    await check_rate_limit(request, api_key)

    # Cache check
    cache = QueryCache()
    config_dict = body.retrieval_config.model_dump() if body.retrieval_config else None
    cached = await cache.get(
        body.question, tenant_id, body.document_ids, config_dict
    )
    if cached and not body.enable_developer_mode:
        logger.info("Cache HIT for query")
        result = QueryAnswer.model_validate(cached)
        result.cache_hit = True
        return result

    # Run pipeline
    answer = await _run_pipeline(
        question=body.question,
        tenant_id=tenant_id,
        document_ids=body.document_ids,
        conversation_id=body.conversation_id,
        enable_developer_mode=body.enable_developer_mode,
        retrieval_config=body.retrieval_config,
    )

    # Cache the result
    if not body.enable_developer_mode:
        await cache.set(
            body.question,
            tenant_id,
            answer.model_dump(),
            body.document_ids,
            config_dict,
        )

    return answer


@router.post("/stream")
async def query_stream(
    request: Request,
    body: QueryRequest,
    _: str = Depends(require_api_key),
):
    """
    Stream query response with Server-Sent Events.
    Status events are sent during pipeline execution, then the answer is streamed.
    """
    tenant_id = get_tenant_id(request)

    async def _event_generator() -> AsyncGenerator[str, None]:
        # Status updates during pipeline
        status_events = [
            "Analyzing query...",
            "Rewriting query...",
            "Searching documents...",
            "Running hybrid retrieval...",
            "Reranking results...",
            "Compressing context...",
            "Generating answer...",
            "Verifying evidence...",
        ]

        for status_msg in status_events:
            if await request.is_disconnected():
                return
            yield f"data: {json.dumps({'type': 'status', 'content': status_msg})}\n\n"

        try:
            answer = await _run_pipeline(
                question=body.question,
                tenant_id=tenant_id,
                document_ids=body.document_ids,
                conversation_id=body.conversation_id,
                enable_developer_mode=body.enable_developer_mode,
                retrieval_config=body.retrieval_config,
            )

            # Stream the answer word by word
            words = answer.answer.split(" ")
            for word in words:
                if await request.is_disconnected():
                    return
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

            # Send final metadata
            yield f"data: {json.dumps({'type': 'done', 'metadata': answer.model_dump(mode='json')})}\n\n"

        except Exception as exc:
            logger.error("Streaming query failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{query_id}/evidence")
async def get_evidence(
    query_id: str,
    request: Request,
    _: str = Depends(require_api_key),
):
    """Retrieve the evidence and pipeline data for a previous query."""
    tenant_id = get_tenant_id(request)
    redis = get_redis_client()

    raw = await redis.get(f"answer:{tenant_id}:{query_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Query result not found or expired")

    return QueryAnswer.model_validate_json(raw)


@router.get("/stats/summary", response_model=SystemStats)
async def get_stats(
    request: Request,
    _: str = Depends(require_api_key),
):
    """Return system statistics for the tenant dashboard."""
    tenant_id = get_tenant_id(request)
    redis = get_redis_client()

    # Count documents
    doc_ids = await redis.smembers(f"tenant_docs:{tenant_id}")
    total_docs = len(doc_ids)

    total_pages = 0
    total_chunks = 0
    for doc_id in doc_ids:
        raw = await redis.get(f"doc_meta:{tenant_id}:{doc_id}")
        if raw:
            from app.models.schemas import DocumentMetadata
            meta = DocumentMetadata.model_validate_json(raw)
            total_pages += meta.pages
            total_chunks += meta.chunks

    # Query stats
    query_count = int(await redis.get(f"stats:queries:{tenant_id}") or 0)
    avg_latency = float(await redis.get(f"stats:avg_latency:{tenant_id}") or 0)
    cache_hits = int(await redis.get(f"stats:cache_hits:{tenant_id}") or 0)
    cache_hit_rate = cache_hits / max(query_count, 1)

    return SystemStats(
        total_documents=total_docs,
        total_pages=total_pages,
        total_chunks=total_chunks,
        total_queries=query_count,
        avg_latency_ms=avg_latency,
        cache_hit_rate=cache_hit_rate,
        tenant_id=tenant_id,
    )
