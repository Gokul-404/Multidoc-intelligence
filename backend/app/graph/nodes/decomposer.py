"""
Query Decomposer node for SentinelRAG LangGraph pipeline.
Breaks complex multi-hop questions into sub-questions.
"""
from __future__ import annotations

import logging
import time

from app.graph.state import GraphState
from app.langchain_components.models import get_model_service
from app.langchain_components.prompts import QUERY_DECOMPOSITION_PROMPT
from app.models.schemas import PipelineStage

logger = logging.getLogger(__name__)


def decompose_query(state: GraphState) -> GraphState:
    """
    Node: Decompose complex questions into independent sub-questions.
    Only runs if query_analysis indicates decomposition is needed.
    """
    t0 = time.monotonic()
    analysis = state.get("query_analysis")
    question = state.get("question", "")

    # Only decompose when flagged by the analyzer
    if not (analysis and analysis.requires_decomposition):
        return {**state, "subqueries": [question]}

    model = get_model_service()
    subqueries: list[str] = []

    try:
        prompt = QUERY_DECOMPOSITION_PROMPT.format_messages(question=question)
        raw = model.generate(prompt)
        parsed = model.parse_json_response(raw)
        if isinstance(parsed, list):
            subqueries = [str(q).strip() for q in parsed if str(q).strip()]
    except Exception as exc:
        logger.warning("Query decomposition failed: %s", exc)

    if not subqueries:
        subqueries = [question]

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info("Decomposed into %d sub-questions in %.0fms", len(subqueries), latency_ms)

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="query_decomposition",
            data={"original": question, "subqueries": subqueries},
            latency_ms=latency_ms,
        ))

    return {**state, "subqueries": subqueries, "pipeline_stages": stages}
