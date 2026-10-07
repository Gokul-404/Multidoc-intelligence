"""
Evidence Auditor node for SentinelRAG LangGraph pipeline.
Runs the full audit after generation and manages the retry/refusal loop.
"""
from __future__ import annotations

import logging
import time

from app.audit.auditor import EvidenceAuditor
from app.graph.state import GraphState
from app.langchain_components.models import get_model_service
from app.langchain_components.prompts import QUERY_IMPROVEMENT_PROMPT
from app.models.schemas import AuditStatus, PipelineStage

logger = logging.getLogger(__name__)

_auditor = EvidenceAuditor()
REFUSAL_MESSAGE = "I couldn't verify this from the uploaded documents."


def audit_evidence(state: GraphState) -> GraphState:
    """
    Node: Run full evidence audit (claims, citations, numbers, contradictions).
    Updates audit_result in state.
    """
    t0 = time.monotonic()
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    compressed_docs = state.get("compressed_docs", [])
    question = state.get("question", "")
    reranked = state.get("reranked_results", [])

    # Build max_page_by_doc from reranked results
    max_page_by_doc: dict[str, int] = {}
    for chunk in reranked:
        doc_id = chunk.document.metadata.get("document_id", "")
        page = chunk.document.metadata.get("page", 0)
        if doc_id:
            max_page_by_doc[doc_id] = max(max_page_by_doc.get(doc_id, 0), page * 10)  # Generous upper bound

    reranker_scores = [c.score for c in reranked]

    audit_result = _auditor.audit(
        answer=answer,
        citations=citations,
        evidence_docs=compressed_docs,
        question=question,
        max_page_by_doc=max_page_by_doc,
        reranker_scores=reranker_scores,
    )

    latency_ms = (time.monotonic() - t0) * 1000

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="evidence_audit",
            data={
                "status": audit_result.status.value,
                "claims_total": audit_result.claims_total,
                "claims_supported": audit_result.claims_supported,
                "claims_unsupported": audit_result.claims_unsupported,
                "citations_valid": audit_result.citations_valid,
                "citations_invalid": audit_result.citations_invalid,
                "numerical_verified": audit_result.numerical_claims_verified,
                "has_contradictions": audit_result.has_contradictions,
                "confidence": round(audit_result.confidence_score, 1),
                "failure_reasons": audit_result.failure_reasons,
            },
            latency_ms=latency_ms,
        ))

    return {**state, "audit_result": audit_result, "pipeline_stages": stages}


def improve_query_after_failure(state: GraphState) -> GraphState:
    """
    Node: After audit failure, generate an improved query for retry.
    This implements the self-correction mechanism.
    """
    audit_result = state.get("audit_result")
    question = state.get("question", "")
    failure_reasons = audit_result.failure_reasons if audit_result else ["Unknown failure"]

    model = get_model_service()
    try:
        prompt = QUERY_IMPROVEMENT_PROMPT.format_messages(
            question=question,
            failure_reasons="; ".join(failure_reasons),
        )
        improved = model.generate(prompt).strip()
        if improved and len(improved) > 5:
            logger.info("Self-correction: improved query: %.80s", improved)
        else:
            improved = question
    except Exception as exc:
        logger.warning("Query improvement failed: %s", exc)
        improved = question

    retry_count = state.get("retry_count", 0) + 1
    return {
        **state,
        "question": improved,
        "rewritten_queries": [improved],
        "subqueries": [improved],
        "retry_count": retry_count,
    }


def refuse_answer(state: GraphState) -> GraphState:
    """
    Node: Issue a safe refusal after max retries exhausted.
    """
    audit_result = state.get("audit_result")
    reasons = audit_result.failure_reasons if audit_result else []
    logger.info("Refusing answer after %d retries. Reasons: %s", state.get("retry_count", 0), reasons)
    return {
        **state,
        "answer": REFUSAL_MESSAGE,
        "citations": [],
        "refused": True,
    }
