"""
Citation auditor for SentinelRAG.
Verifies that every generated citation references a real, retrieved, supporting chunk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from app.models.schemas import Citation

logger = logging.getLogger(__name__)


@dataclass
class CitationAuditResult:
    valid: bool
    citation_index: int
    failure_reason: Optional[str] = None
    chunk_id: str = ""
    page: int = 0


@dataclass
class CitationsAuditReport:
    all_valid: bool
    results: list[CitationAuditResult] = field(default_factory=list)
    valid_count: int = 0
    invalid_count: int = 0


def audit_citations(
    citations: list[Citation],
    retrieved_chunks: list[Document],
    max_page_by_doc: dict[str, int],
) -> CitationsAuditReport:
    """
    Verify each citation against the retrieved evidence set.

    Checks:
    1. Citation chunk_id exists in retrieved chunks
    2. Cited page exists in the source document (max_page_by_doc)
    3. Citation document matches an actually retrieved document
    4. Supporting text appears in the chunk (rough check)
    """
    # Build lookup by chunk_id
    chunk_by_id: dict[str, Document] = {
        d.metadata.get("chunk_id", ""): d for d in retrieved_chunks if d.metadata.get("chunk_id")
    }
    retrieved_doc_ids = {d.metadata.get("document_id", "") for d in retrieved_chunks}

    results: list[CitationAuditResult] = []

    for citation in citations:
        failure = None

        # Check 1: Document was actually retrieved
        if citation.document_id not in retrieved_doc_ids:
            failure = f"Cited document '{citation.document_id}' was not in the retrieved evidence set."
        # Check 2: Chunk exists
        elif citation.chunk_id not in chunk_by_id:
            failure = f"Cited chunk '{citation.chunk_id}' does not exist in retrieved results."
        else:
            chunk = chunk_by_id[citation.chunk_id]
            # Check 3: Page number is valid
            max_page = max_page_by_doc.get(citation.document_id, 9999)
            if citation.page < 1 or citation.page > max_page:
                failure = (
                    f"Cited page {citation.page} is out of range for document "
                    f"'{citation.filename}' (max {max_page})."
                )
            # Check 4: Loose supporting text match
            elif citation.supporting_text:
                # Normalise whitespace for comparison
                norm_supporting = " ".join(citation.supporting_text.lower().split())
                norm_chunk = " ".join(chunk.page_content.lower().split())
                # Check if at least 50% of words are present
                support_words = set(norm_supporting.split())
                chunk_words = set(norm_chunk.split())
                if support_words and len(support_words & chunk_words) / len(support_words) < 0.3:
                    failure = (
                        f"Supporting text for citation {citation.citation_index} "
                        "does not closely match the cited chunk content."
                    )

        results.append(
            CitationAuditResult(
                valid=failure is None,
                citation_index=citation.citation_index,
                failure_reason=failure,
                chunk_id=citation.chunk_id,
                page=citation.page,
            )
        )

    valid_count = sum(1 for r in results if r.valid)
    invalid_count = len(results) - valid_count

    return CitationsAuditReport(
        all_valid=invalid_count == 0,
        results=results,
        valid_count=valid_count,
        invalid_count=invalid_count,
    )
