"""
LangGraph StateGraph workflow for SentinelRAG.

Pipeline:
    analyze → route → [rewrite] → [decompose] → retrieve → rerank → parent → compress → generate → audit
                                                                                               ↑          ↓
                                                                                          retry ← fail  pass → END
                                                                                            ↓ (max 2)
                                                                                          refuse → END

Uses:
- StateGraph for stateful orchestration
- Conditional edges for routing and retry logic
- Typed GraphState for all nodes
- MAX_RETRIES = 2 enforced in conditional routing
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.graph.nodes.analyzer import analyze_query
from app.graph.nodes.auditor import audit_evidence, improve_query_after_failure, refuse_answer
from app.graph.nodes.compressor import compress_context
from app.graph.nodes.decomposer import decompose_query
from app.graph.nodes.generator import generate_answer
from app.graph.nodes.reranker import rerank_results, retrieve_parents
from app.graph.nodes.retriever import retrieve_documents
from app.graph.nodes.rewriter import rewrite_query
from app.graph.state import GraphState
from app.models.schemas import AuditStatus, QueryRoute

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Routing functions ─────────────────────────────────────────────────────────

def route_query(state: GraphState) -> Literal["rewrite", "end_unsupported"]:
    """
    Conditional edge: Route query based on analysis.
    Unsupported queries are immediately refused.
    """
    route = state.get("route", QueryRoute.SIMPLE)
    if route == QueryRoute.UNSUPPORTED:
        logger.info("Query routed to UNSUPPORTED; refusing immediately")
        return "end_unsupported"
    return "rewrite"


def should_decompose(state: GraphState) -> Literal["decompose", "retrieve"]:
    """Conditional edge: decompose only if flagged by analysis."""
    analysis = state.get("query_analysis")
    if analysis and analysis.requires_decomposition:
        return "decompose"
    return "retrieve"


def audit_decision(
    state: GraphState,
) -> Literal["pass", "retry", "refuse"]:
    """
    Conditional edge: Check audit result and retry count.
    MAX_RETRIES enforced here. If exceeded, refuse.
    """
    audit_result = state.get("audit_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)
    refused = state.get("refused", False)

    # If already refused or no evidence
    if refused:
        return "pass"  # Refusals are valid responses

    if audit_result and audit_result.status == AuditStatus.PASS:
        return "pass"

    # FAIL path
    if retry_count >= max_retries:
        logger.warning("Max retries (%d) exhausted; refusing", max_retries)
        return "refuse"

    return "retry"


def handle_unsupported(state: GraphState) -> GraphState:
    """Immediately refuse unsupported query types."""
    return {
        **state,
        "answer": "I couldn't verify this from the uploaded documents. The query type is not supported.",
        "citations": [],
        "refused": True,
    }


# ── Graph construction ────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    Build and compile the SentinelRAG LangGraph StateGraph.
    """
    workflow = StateGraph(GraphState)

    # Register all nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("route", lambda s: s)  # Routing is done via conditional edge
    workflow.add_node("handle_unsupported", handle_unsupported)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("decompose", decompose_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank", rerank_results)
    workflow.add_node("parent_retrieve", retrieve_parents)
    workflow.add_node("compress", compress_context)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("audit", audit_evidence)
    workflow.add_node("improve_query", improve_query_after_failure)
    workflow.add_node("refuse", refuse_answer)

    # Entry point
    workflow.set_entry_point("analyze")

    # analyze → conditional routing
    workflow.add_conditional_edges(
        "analyze",
        route_query,
        {
            "rewrite": "rewrite",
            "end_unsupported": "handle_unsupported",
        },
    )

    # handle_unsupported → END
    workflow.add_edge("handle_unsupported", END)

    # rewrite → conditional decompose or retrieve
    workflow.add_conditional_edges(
        "rewrite",
        should_decompose,
        {
            "decompose": "decompose",
            "retrieve": "retrieve",
        },
    )

    # decompose → retrieve
    workflow.add_edge("decompose", "retrieve")

    # retrieve → rerank → parent → compress → generate → audit
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "parent_retrieve")
    workflow.add_edge("parent_retrieve", "compress")
    workflow.add_edge("compress", "generate")
    workflow.add_edge("generate", "audit")

    # audit → conditional: pass | retry | refuse
    workflow.add_conditional_edges(
        "audit",
        audit_decision,
        {
            "pass": END,
            "retry": "improve_query",
            "refuse": "refuse",
        },
    )

    # retry loop: improve_query → retrieve (skip analysis/rewrite on retry)
    workflow.add_edge("improve_query", "retrieve")

    # refuse → END
    workflow.add_edge("refuse", END)

    return workflow.compile()


# Compiled graph singleton
_compiled_graph = None


def get_workflow():
    """Get or build the compiled LangGraph workflow."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_workflow()
        logger.info("LangGraph workflow compiled successfully")
    return _compiled_graph
