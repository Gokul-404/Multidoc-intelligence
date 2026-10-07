"""
Answer Generator node for SentinelRAG LangGraph pipeline.
Generates grounded answers using LangChain prompt templates.
Prompt injection is mitigated by wrapping document content in delimiters.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from langchain_core.documents import Document

from app.graph.state import GraphState
from app.langchain_components.models import get_model_service
from app.langchain_components.prompts import (
    ANSWER_GENERATION_PROMPT,
    DOCUMENT_WRAPPER_END,
    DOCUMENT_WRAPPER_START,
)
from app.models.schemas import Citation, PipelineStage

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I couldn't verify this from the uploaded documents."


def _format_evidence_for_generation(docs: list[Document]) -> str:
    """
    Format retrieved documents for the answer generation prompt.
    Each document is wrapped in injection-resistant delimiters.
    Document content is clearly marked as DATA, not instructions.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section") or ""
        location = f"{filename} — p.{page}"
        if section:
            location += f" [{section}]"

        # Wrap content in injection-resistant delimiters
        parts.append(
            f"[{i}] chunk_id={chunk_id} | source={location}\n"
            f"{DOCUMENT_WRAPPER_START}\n{doc.page_content}\n{DOCUMENT_WRAPPER_END}"
        )
    return "\n\n".join(parts)


def _build_citations(
    citations_used: list[int],
    docs: list[Document],
    reranked_chunks: list,
) -> list[Citation]:
    """Build structured Citation objects from citation indices."""
    # Build score map from reranked chunks
    score_map = {c.chunk_id: c.score for c in (reranked_chunks or [])}

    citations = []
    for idx in citations_used:
        doc_index = idx - 1  # Convert 1-indexed to 0-indexed
        if 0 <= doc_index < len(docs):
            doc = docs[doc_index]
            chunk_id = doc.metadata.get("chunk_id", "")
            citations.append(
                Citation(
                    citation_index=idx,
                    document_id=doc.metadata.get("document_id", ""),
                    filename=doc.metadata.get("filename", "unknown"),
                    page=doc.metadata.get("page", 0),
                    chunk_id=chunk_id,
                    supporting_text=doc.page_content[:200],
                    retrieval_score=score_map.get(chunk_id),
                )
            )
    return citations


def generate_answer(state: GraphState) -> GraphState:
    """
    Node: Generate a grounded answer using the compressed evidence.
    Treats all document content as untrusted data.
    Returns answer, citations, and whether the query was refused.
    """
    t0 = time.monotonic()
    compressed_docs = state.get("compressed_docs", [])
    reranked_results = state.get("reranked_results", [])
    question = state.get("question", "")
    conversation_context = state.get("conversation_context", "")
    model = get_model_service()

    if not compressed_docs:
        logger.warning("No evidence available; refusing query")
        return {
            **state,
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "refused": True,
            "answer_id": str(uuid.uuid4()),
        }

    formatted_evidence = _format_evidence_for_generation(compressed_docs)

    try:
        prompt = ANSWER_GENERATION_PROMPT.format_messages(
            conversation_context=conversation_context or "No prior conversation.",
            question=question,
            formatted_evidence=formatted_evidence,
        )
        raw = model.generate(prompt)
        parsed = model.parse_json_response(raw)

        if parsed and isinstance(parsed, dict):
            answer_text = parsed.get("answer", REFUSAL_MESSAGE)
            citations_used = parsed.get("citations_used", [])
            refused = bool(parsed.get("refused", False))

            # Build citations
            citations = _build_citations(citations_used, compressed_docs, reranked_results)
        else:
            answer_text = REFUSAL_MESSAGE
            citations = []
            refused = True

    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        answer_text = REFUSAL_MESSAGE
        citations = []
        refused = True

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info("Generated answer (%.0f chars) in %.0fms", len(answer_text), latency_ms)

    stages = list(state.get("pipeline_stages", []))
    if state.get("enable_developer_mode"):
        stages.append(PipelineStage(
            stage="generation",
            data={
                "answer_length": len(answer_text),
                "citations_count": len(citations),
                "refused": refused,
                "evidence_chunks_used": len(compressed_docs),
            },
            latency_ms=latency_ms,
        ))

    return {
        **state,
        "answer": answer_text,
        "citations": citations,
        "refused": refused,
        "answer_id": str(uuid.uuid4()),
        "pipeline_stages": stages,
    }
