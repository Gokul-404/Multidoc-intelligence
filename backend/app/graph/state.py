"""
LangGraph typed state for SentinelRAG query pipeline.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict
import operator

from langchain_core.documents import Document

from app.models.schemas import (
    AuditResult,
    Citation,
    PipelineStage,
    QueryAnalysis,
    QueryRoute,
    RetrievalConfig,
)
from app.retrieval.qdrant import RetrievedChunk


class GraphState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    question: str
    original_question: str
    tenant_id: str
    document_ids: Optional[list[str]]
    conversation_id: Optional[str]
    conversation_context: str
    retrieval_config: Optional[RetrievalConfig]
    enable_developer_mode: bool

    # ── Query analysis ─────────────────────────────────────────────────────────
    query_analysis: Optional[QueryAnalysis]
    route: QueryRoute
    rewritten_queries: list[str]
    subqueries: list[str]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    dense_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    hybrid_results: list[RetrievedChunk]
    reranked_results: list[RetrievedChunk]
    parent_docs: list[Document]
    compressed_docs: list[Document]

    # ── Retrieval metrics ─────────────────────────────────────────────────────
    dense_latency_ms: float
    bm25_latency_ms: float
    rerank_latency_ms: float
    compress_latency_ms: float
    tokens_before_compression: int
    tokens_after_compression: int

    # ── Generation ─────────────────────────────────────────────────────────────
    answer: str
    citations: list[Citation]
    refused: bool

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_result: Optional[AuditResult]
    retry_count: int
    max_retries: int

    # ── Developer mode ────────────────────────────────────────────────────────
    pipeline_stages: list[PipelineStage]

    # ── Errors ────────────────────────────────────────────────────────────────
    error: Optional[str]
    answer_id: str
