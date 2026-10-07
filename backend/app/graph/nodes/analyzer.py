"""
Query Analyzer node for SentinelRAG LangGraph pipeline.
Uses LangChain structured output + Pydantic to classify the query.
"""
from __future__ import annotations

import logging
import time

from app.graph.state import GraphState
from app.langchain_components.models import get_model_service
from app.langchain_components.prompts import QUERY_ANALYSIS_PROMPT
from app.models.schemas import PipelineStage, QueryAnalysis, QueryRoute

logger = logging.getLogger(__name__)


def analyze_query(state: GraphState) -> GraphState:
    """
    Node: Analyse the user's question to determine intent, complexity,
    required entities, and optimal query route.
    """
    t0 = time.monotonic()
    question = state.get("question", "")
    model = get_model_service()

    analysis = QueryAnalysis(
        intent="factual",
        complexity="simple",
        route=QueryRoute.SIMPLE,
    )

    try:
        prompt = QUERY_ANALYSIS_PROMPT.format_messages(question=question)
        raw = model.generate(prompt)
        parsed = model.parse_json_response(raw)

        if parsed and isinstance(parsed, dict):
            route_str = parsed.get("route", "simple")
            try:
                route = QueryRoute(route_str)
            except ValueError:
                route = QueryRoute.SIMPLE

            analysis = QueryAnalysis(
                intent=parsed.get("intent", "factual"),
                complexity=parsed.get("complexity", "simple"),
                entities=parsed.get("entities", []),
                keywords=parsed.get("keywords", []),
                document_constraints=parsed.get("document_constraints", []),
                requires_multi_document=parsed.get("requires_multi_document", False),
                requires_decomposition=parsed.get("requires_decomposition", False),
                requires_numerical_reasoning=parsed.get("requires_numerical_reasoning", False),
                route=route,
            )
    except Exception as exc:
        logger.warning("Query analysis failed: %s; using defaults", exc)

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info("Query analysis: route=%s complexity=%s in %.0fms", analysis.route, analysis.complexity, latency_ms)

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="query_analysis",
            data={
                "intent": analysis.intent,
                "complexity": analysis.complexity,
                "route": analysis.route.value,
                "entities": analysis.entities,
                "keywords": analysis.keywords,
                "requires_multi_document": analysis.requires_multi_document,
                "requires_decomposition": analysis.requires_decomposition,
                "requires_numerical_reasoning": analysis.requires_numerical_reasoning,
            },
            latency_ms=latency_ms,
        ))

    return {
        **state,
        "query_analysis": analysis,
        "route": analysis.route,
        "pipeline_stages": stages,
    }
