"""
Retriever node for SentinelRAG LangGraph pipeline.
Runs multi-query hybrid retrieval using all rewritten queries and sub-questions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from langchain_core.documents import Document

from app.graph.state import GraphState
from app.models.schemas import PipelineStage, RetrievalConfig
from app.retrieval.hybrid import HybridRetriever, HybridResult, reciprocal_rank_fusion
from app.retrieval.qdrant import RetrievedChunk

logger = logging.getLogger(__name__)


def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate by chunk_id, keeping highest score."""
    seen: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        cid = chunk.chunk_id
        if cid not in seen or chunk.score > seen[cid].score:
            seen[cid] = chunk
    return list(seen.values())


async def _run_multi_query_retrieval(
    queries: list[str],
    tenant_id: str,
    document_ids: Optional[list[str]],
    config: Optional[RetrievalConfig],
) -> tuple[list[RetrievedChunk], float, float, int, int]:
    """
    Run hybrid retrieval for each query in parallel, then fuse results.
    Returns (fused_chunks, dense_latency_ms, bm25_latency_ms, dense_count, bm25_count)
    """
    retriever = HybridRetriever(
        dense_weight=config.dense_weight if config else None,
        bm25_weight=config.bm25_weight if config else None,
    )

    tasks = [
        retriever.retrieve(
            query=q,
            tenant_id=tenant_id,
            document_ids=document_ids,
            dense_k=config.dense_top_k if config else None,
            bm25_k=config.bm25_top_k if config else None,
        )
        for q in queries
    ]

    results: list[HybridResult] = await asyncio.gather(*tasks)

    # Collect all candidate lists for final RRF
    all_dense: list[RetrievedChunk] = []
    all_bm25: list[RetrievedChunk] = []
    all_fused_lists: list[list[RetrievedChunk]] = []
    total_dense_latency = 0.0
    total_bm25_latency = 0.0

    for hr in results:
        all_fused_lists.append(hr.chunks)
        total_dense_latency = max(total_dense_latency, hr.dense_latency_ms)
        total_bm25_latency = max(total_bm25_latency, hr.bm25_latency_ms)

    # Final cross-query RRF
    final_fused = reciprocal_rank_fusion(all_fused_lists)
    final_fused = _deduplicate(final_fused)

    dense_count = sum(r.dense_count for r in results)
    bm25_count = sum(r.bm25_count for r in results)

    return final_fused, total_dense_latency, total_bm25_latency, dense_count, bm25_count


def retrieve_documents(state: GraphState) -> GraphState:
    """
    Node: Multi-query hybrid retrieval.
    Retrieves candidates for all rewritten queries and sub-questions,
    then deduplicates and fuses with RRF.
    """
    t0 = time.monotonic()
    tenant_id = state.get("tenant_id", "default")
    document_ids = state.get("document_ids")
    config = state.get("retrieval_config")

    # Combine all query variants
    queries = list(set(
        state.get("rewritten_queries", []) +
        state.get("subqueries", []) +
        [state.get("question", "")]
    ))
    queries = [q for q in queries if q.strip()][:8]  # Cap at 8 to avoid overloading

    try:
        fused, dense_latency, bm25_latency, dense_count, bm25_count = asyncio.get_event_loop().run_until_complete(
            _run_multi_query_retrieval(queries, tenant_id, document_ids, config)
        )
    except RuntimeError:
        # If event loop is already running (e.g., in async context)
        import nest_asyncio
        nest_asyncio.apply()
        fused, dense_latency, bm25_latency, dense_count, bm25_count = asyncio.get_event_loop().run_until_complete(
            _run_multi_query_retrieval(queries, tenant_id, document_ids, config)
        )

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Multi-query retrieval: %d queries → %d unique candidates in %.0fms",
        len(queries), len(fused), latency_ms,
    )

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="retrieval",
            data={
                "queries": queries,
                "dense_candidates": dense_count,
                "bm25_candidates": bm25_count,
                "unique_after_fusion": len(fused),
                "dense_latency_ms": round(dense_latency, 1),
                "bm25_latency_ms": round(bm25_latency, 1),
            },
            latency_ms=latency_ms,
        ))

    return {
        **state,
        "hybrid_results": fused,
        "dense_latency_ms": dense_latency,
        "bm25_latency_ms": bm25_latency,
        "pipeline_stages": stages,
    }
