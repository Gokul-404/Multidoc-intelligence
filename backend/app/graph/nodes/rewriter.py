"""
Query Rewriter node for SentinelRAG LangGraph pipeline.
Generates retrieval-optimised variants of the user query.
"""
from __future__ import annotations

import logging
import time

from app.graph.state import GraphState
from app.langchain_components.models import get_model_service
from app.langchain_components.prompts import QUERY_REWRITE_PROMPT
from app.models.schemas import PipelineStage, QueryRoute

logger = logging.getLogger(__name__)

# Number of rewrite variants to generate
_N_REWRITES = 4


def rewrite_query(state: GraphState) -> GraphState:
    """
    Node: Generate multiple retrieval-optimised query variants.
    Converts vague queries into retrieval-friendly keywords while preserving intent.
    """
    t0 = time.monotonic()
    question = state.get("question", "")
    model = get_model_service()
    rewritten: list[str] = [question]  # Always include original

    try:
        prompt = QUERY_REWRITE_PROMPT.format_messages(question=question, n=_N_REWRITES)
        raw = model.generate(prompt)
        parsed = model.parse_json_response(raw)
        if isinstance(parsed, list):
            extras = [str(q).strip() for q in parsed if str(q).strip()]
            # Deduplicate while preserving order
            seen = {question.lower()}
            for q in extras:
                if q.lower() not in seen:
                    rewritten.append(q)
                    seen.add(q.lower())
    except Exception as exc:
        logger.warning("Query rewriting failed: %s", exc)

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info("Rewrote query into %d variants in %.0fms", len(rewritten), latency_ms)

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="query_rewrite",
            data={"original": question, "rewrites": rewritten},
            latency_ms=latency_ms,
        ))

    return {**state, "rewritten_queries": rewritten, "pipeline_stages": stages}
