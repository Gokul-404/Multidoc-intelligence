"""
Document metadata enrichment for SentinelRAG.
Infers document type, language, author, and creation date.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.models.schemas import DocumentType

logger = logging.getLogger(__name__)

# ── Document type detection ───────────────────────────────────────────────────

_TYPE_SIGNALS: dict[DocumentType, list[str]] = {
    DocumentType.RESEARCH_PAPER: [
        "abstract", "introduction", "methodology", "related work",
        "conclusion", "references", "arxiv", "doi", "journal",
        "proceedings", "conference", "hypothesis", "experiment",
    ],
    DocumentType.FINANCIAL_REPORT: [
        "revenue", "ebitda", "earnings per share", "fiscal year",
        "balance sheet", "income statement", "cash flow", "quarterly",
        "annual report", "10-k", "10-q", "financial statements",
    ],
    DocumentType.LEGAL_DOCUMENT: [
        "whereas", "hereinafter", "notwithstanding", "indemnify",
        "jurisdiction", "pursuant", "liability", "agreement", "contract",
        "clause", "covenant", "party", "obligations", "warranty",
    ],
    DocumentType.TECHNICAL_DOCUMENT: [
        "api", "endpoint", "implementation", "architecture", "deployment",
        "configuration", "installation", "requirements", "dependencies",
        "documentation", "specification", "system design", "module",
    ],
    DocumentType.RESUME: [
        "curriculum vitae", "work experience", "education", "skills",
        "certifications", "objective", "summary", "employment history",
        "references available", "linkedin",
    ],
    DocumentType.TEXTBOOK: [
        "chapter", "exercise", "learning objectives", "summary",
        "further reading", "definition", "theorem", "proof",
        "example", "figure", "table of contents",
    ],
}


def detect_document_type(text_sample: str) -> DocumentType:
    """
    Detect document type from the first ~3000 characters of text.
    Returns the best match or GENERAL if nothing matches well.
    """
    sample = text_sample[:3000].lower()
    scores: dict[DocumentType, int] = {dt: 0 for dt in DocumentType}

    for doc_type, signals in _TYPE_SIGNALS.items():
        for signal in signals:
            if signal in sample:
                scores[doc_type] += 1

    best_type = max(scores, key=lambda dt: scores[dt])
    if scores[best_type] >= 2:
        logger.debug("Detected document type: %s (score=%d)", best_type, scores[best_type])
        return best_type
    return DocumentType.GENERAL


def detect_language(text_sample: str) -> str:
    """
    Simple language detection using langdetect.
    Falls back to 'en' on error.
    """
    try:
        from langdetect import detect
        lang = detect(text_sample[:1000])
        return lang or "en"
    except Exception:
        return "en"


def extract_pdf_metadata(pdf_bytes: bytes) -> dict:
    """Extract metadata from PDF XMP/Info dictionary."""
    import io
    import pypdf

    result = {"author": None, "creation_date": None, "title": None}
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes), strict=False)
        info = reader.metadata
        if info:
            result["author"] = getattr(info, "author", None) or info.get("/Author")
            result["title"] = getattr(info, "title", None) or info.get("/Title")
            creation = getattr(info, "creation_date", None) or info.get("/CreationDate")
            if creation:
                result["creation_date"] = str(creation)[:10]  # YYYY-MM-DD
    except Exception as exc:
        logger.debug("Could not extract PDF metadata: %s", exc)
    return result


def build_chunk_metadata(
    document_id: str,
    filename: str,
    tenant_id: str,
    page: int,
    section: Optional[str],
    chunk_index: int,
    parent_id: Optional[str],
    chunk_type: str,
    document_type: DocumentType,
    language: str = "en",
    author: Optional[str] = None,
    creation_date: Optional[str] = None,
) -> dict:
    """Construct canonical metadata dict for a LangChain Document chunk."""
    chunk_id = f"{document_id}_pg{page}_ch{chunk_index}"
    if parent_id:
        chunk_id = f"{parent_id}_c{chunk_index}"

    return {
        "document_id": document_id,
        "filename": filename,
        "tenant_id": tenant_id,
        "page": page,
        "section": section,
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "chunk_type": chunk_type,
        "document_type": document_type.value,
        "language": language,
        "author": author,
        "creation_date": creation_date,
    }
