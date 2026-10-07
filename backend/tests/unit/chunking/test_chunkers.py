"""
Unit tests for SentinelRAG text splitters and chunking strategies.
"""
import pytest
from langchain_core.documents import Document

from app.langchain_components.splitters import (
    build_parent_child_chunks,
    get_parent_splitter,
    get_recursive_splitter,
    get_structure_aware_splitter,
)


def test_recursive_splitter_chunk_size_respect():
    splitter = get_recursive_splitter(chunk_size=100, chunk_overlap=20)
    text = "Paragraph one with several sentences. " * 10
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120  # Allowing separator margin


def test_parent_child_chunks_hierarchy():
    text = (
        "Section 1: Market Analysis. In 2024 the global cloud market expanded significantly. "
        "Enterprise adoption accelerated across EMEA and APAC regions. "
        "Section 2: Financial Performance. Revenue reached $42.5 billion, up 22 percent YoY. "
        "Operating profit margins reached 34.2 percent."
    )
    doc = Document(
        page_content=text,
        metadata={"document_id": "doc_fin_001", "page": 1, "filename": "report.pdf"},
    )

    parents, children = build_parent_child_chunks([doc])

    assert len(parents) >= 1
    assert len(children) >= len(parents)

    # Check child references parent_id properly
    parent_ids = {p.metadata["chunk_id"] for p in parents}
    for child in children:
        assert "parent_id" in child.metadata
        assert child.metadata["parent_id"] in parent_ids
        assert child.metadata["document_id"] == "doc_fin_001"
        assert child.metadata["chunk_type"] == "child"


def test_structure_aware_splitter_sections():
    markdown_doc = (
        "# Executive Summary\n\n"
        "This project delivers AI grounded multi-agent RAG.\n\n"
        "# Architecture\n\n"
        "Built on LangChain, LangGraph, Qdrant, and BM25.\n\n"
        "## Components\n\n"
        "Hierarchical chunker, RRF fusion, Cross-encoder reranking."
    )
    splitter = get_structure_aware_splitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(markdown_doc)
    assert len(chunks) >= 2
    # Check that headers are preserved
    assert any("Executive Summary" in c for c in chunks)
    assert any("Architecture" in c for c in chunks)
