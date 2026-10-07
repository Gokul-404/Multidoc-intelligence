"""
Unit tests for citation auditing.
"""
import pytest
from langchain_core.documents import Document
from app.audit.citations import audit_citations, CitationsAuditReport
from app.models.schemas import Citation


def make_doc(chunk_id: str, document_id: str, filename: str, page: int, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "filename": filename,
            "page": page,
            "tenant_id": "tenant1",
        },
    )


def make_citation(
    idx: int, doc_id: str, filename: str, page: int, chunk_id: str, text: str
) -> Citation:
    return Citation(
        citation_index=idx,
        document_id=doc_id,
        filename=filename,
        page=page,
        chunk_id=chunk_id,
        supporting_text=text,
    )


class TestCitationAudit:

    def test_valid_citation_passes(self):
        doc = make_doc("doc1_p0_c0", "doc1", "paper.pdf", 5, "The system uses PostgreSQL.")
        citation = make_citation(1, "doc1", "paper.pdf", 5, "doc1_p0_c0", "The system uses PostgreSQL.")
        max_pages = {"doc1": 100}
        report = audit_citations([citation], [doc], max_pages)
        assert report.all_valid
        assert report.valid_count == 1
        assert report.invalid_count == 0

    def test_fabricated_document_fails(self):
        doc = make_doc("doc1_p0_c0", "doc1", "paper.pdf", 5, "Real content.")
        # Citation references a document not in retrieved set
        citation = make_citation(1, "fakdoc999", "fake.pdf", 5, "doc1_p0_c0", "Content.")
        max_pages = {"doc1": 100}
        report = audit_citations([citation], [doc], max_pages)
        assert not report.all_valid
        assert "not in the retrieved evidence set" in report.results[0].failure_reason

    def test_fabricated_chunk_fails(self):
        doc = make_doc("doc1_p0_c0", "doc1", "paper.pdf", 5, "Real content.")
        citation = make_citation(1, "doc1", "paper.pdf", 5, "NONEXISTENT_CHUNK", "Content.")
        max_pages = {"doc1": 100}
        report = audit_citations([citation], [doc], max_pages)
        assert not report.all_valid

    def test_invalid_page_number_fails(self):
        doc = make_doc("doc1_p0_c0", "doc1", "paper.pdf", 5, "Real content.")
        # Page 999 doesn't exist (max is 100)
        citation = make_citation(1, "doc1", "paper.pdf", 999, "doc1_p0_c0", "Content.")
        max_pages = {"doc1": 100}
        report = audit_citations([citation], [doc], max_pages)
        assert not report.all_valid
        assert "range" in report.results[0].failure_reason

    def test_empty_citations_always_valid(self):
        report = audit_citations([], [], {})
        assert report.all_valid

    def test_multiple_citations_mixed_validity(self):
        doc = make_doc("chunk1", "doc1", "paper.pdf", 3, "PostgreSQL is the database.")
        c1 = make_citation(1, "doc1", "paper.pdf", 3, "chunk1", "PostgreSQL is the database.")
        c2 = make_citation(2, "badoc", "fake.pdf", 1, "fake_chunk", "Made up content.")
        max_pages = {"doc1": 100}
        report = audit_citations([c1, c2], [doc], max_pages)
        assert not report.all_valid
        assert report.valid_count == 1
        assert report.invalid_count == 1
