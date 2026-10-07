"""
Contextual Compression node for SentinelRAG LangGraph pipeline.
"""
from __future__ import annotations

import logging
import time

from app.graph.state import GraphState
from app.langchain_components.compressors import ContextualCompressor
from app.models.schemas import PipelineStage

logger = logging.getLogger(__name__)

_compressor = ContextualCompressor()


def compress_context(state: GraphState) -> GraphState:
    """
    Node: Extract only query-relevant passages from parent context documents.
    Reduces token usage while preserving the key evidence.
    """
    t0 = time.monotonic()
    parent_docs = state.get("parent_docs", [])
    question = state.get("question", "")

    if not parent_docs:
        return {**state, "compressed_docs": [], "tokens_before_compression": 0, "tokens_after_compression": 0}

    result = _compressor.compress(question=question, documents=parent_docs)
    latency_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "Compression: %d → %d tokens (%.0f%%) in %.0fms",
        result.original_tokens,
        result.compressed_tokens,
        result.compression_ratio * 100,
        latency_ms,
    )

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="context_compression",
            data={
                "tokens_before": result.original_tokens,
                "tokens_after": result.compressed_tokens,
                "compression_ratio": round(result.compression_ratio, 3),
                "documents_kept": len(result.documents),
            },
            latency_ms=latency_ms,
        ))

    return {
        **state,
        "compressed_docs": result.documents,
        "tokens_before_compression": result.original_tokens,
        "tokens_after_compression": result.compressed_tokens,
        "compress_latency_ms": latency_ms,
        "pipeline_stages": stages,
    }
