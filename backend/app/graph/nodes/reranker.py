"""
Reranker node + Parent retrieval node for SentinelRAG LangGraph pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.graph.state import GraphState
from app.models.schemas import PipelineStage, RetrievalConfig
from app.retrieval.qdrant import QdrantService, RetrievedChunk
from app.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)
settings = get_settings()

_reranker = Reranker()


def rerank_results(state: GraphState) -> GraphState:
    """
    Node: Cross-encoder reranking of hybrid retrieval candidates.
    Reduces ~20-30 candidates to the top-K most relevant.
    """
    t0 = time.monotonic()
    candidates = state.get("hybrid_results", [])
    config = state.get("retrieval_config")
    question = state.get("question", "")
    top_k = config.rerank_top_k if config else settings.rerank_top_k

    if not candidates:
        return {**state, "reranked_results": [], "rerank_latency_ms": 0.0}

    result = _reranker.rerank(query=question, chunks=candidates, top_k=top_k)
    latency_ms = (time.monotonic() - t0) * 1000

    logger.info("Reranking: %d → %d in %.0fms", len(candidates), len(result.chunks), latency_ms)

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        top_scores = [round(c.score, 4) for c in result.chunks[:5]]
        stages.append(PipelineStage(
            stage="reranking",
            data={
                "input_candidates": result.input_count,
                "output_candidates": result.output_count,
                "top_scores": top_scores,
            },
            latency_ms=latency_ms,
        ))

    return {
        **state,
        "reranked_results": result.chunks,
        "rerank_latency_ms": latency_ms,
        "pipeline_stages": stages,
    }


async def _fetch_parents(
    chunks: list[RetrievedChunk],
    tenant_id: str,
) -> list[Document]:
    """
    Retrieve parent sections for the top reranked child chunks.
    Ensures cross-document parent mapping is prevented by tenant filter.
    """
    qdrant = QdrantService()
    seen_parent_ids: set[str] = set()
    parents: list[Document] = []

    for chunk in chunks:
        parent_id = chunk.document.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        # Prevent cross-document parent mapping:
        # parent must belong to same document
        doc_id = chunk.document.metadata.get("document_id")
        if not parent_id.startswith(doc_id or ""):
            logger.warning(
                "Cross-document parent ID rejected: chunk=%s parent=%s",
                chunk.chunk_id, parent_id,
            )
            continue

        parent = await qdrant.get_parent(parent_id, tenant_id)
        if parent:
            parents.append(parent)
            seen_parent_ids.add(parent_id)
        else:
            # Fallback: use the child chunk itself as context
            parents.append(chunk.document)

    return parents


def retrieve_parents(state: GraphState) -> GraphState:
    """
    Node: Retrieve parent context sections for the top reranked child chunks.
    Provides the LLM with broader surrounding context.
    """
    t0 = time.monotonic()
    reranked = state.get("reranked_results", [])
    tenant_id = state.get("tenant_id", "default")

    if not reranked:
        return {**state, "parent_docs": []}

    try:
        try:
            loop = asyncio.get_event_loop()
            parents = loop.run_until_complete(_fetch_parents(reranked, tenant_id))
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            parents = asyncio.get_event_loop().run_until_complete(_fetch_parents(reranked, tenant_id))
    except Exception as exc:
        logger.error("Parent retrieval failed: %s; falling back to child chunks", exc)
        parents = [c.document for c in reranked]

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info("Retrieved %d parent sections in %.0fms", len(parents), latency_ms)

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="parent_retrieval",
            data={
                "parent_sections": len(parents),
                "source_docs": list({p.metadata.get("filename", "?") for p in parents}),
            },
            latency_ms=latency_ms,
        ))

    return {**state, "parent_docs": parents, "pipeline_stages": stages}
