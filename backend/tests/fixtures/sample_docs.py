"""
Test fixtures and mock factories for SentinelRAG tests.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List
from langchain_core.documents import Document

def make_sample_parent_chunk(
    doc_id: str = "doc_test_123",
    page_number: int = 1,
    text: str = "The quarterly revenue was $15.4 million, representing an 18% YoY increase. Operating margins improved by 250 basis points.",
    chunk_index: int = 0,
) -> Document:
    parent_id = f"parent_{doc_id}_{chunk_index}"
    return Document(
        page_content=text,
        metadata={
            "chunk_id": parent_id,
            "document_id": doc_id,
            "document_name": "Q3_Financial_Report.pdf",
            "page_number": page_number,
            "chunk_type": "parent",
            "token_count": len(text.split()),
            "tenant_id": "default",
        },
    )

def make_sample_child_chunks(
    parent_chunk: Document,
    n_children: int = 2,
) -> List[Document]:
    children = []
    parent_id = parent_chunk.metadata["chunk_id"]
    doc_id = parent_chunk.metadata["document_id"]
    doc_name = parent_chunk.metadata["document_name"]
    page_num = parent_chunk.metadata["page_number"]
    
    words = parent_chunk.page_content.split()
    half = len(words) // 2
    part1 = " ".join(words[:half])
    part2 = " ".join(words[half:])

    children.append(
        Document(
            page_content=part1,
            metadata={
                "chunk_id": f"child_{doc_id}_0",
                "parent_id": parent_id,
                "document_id": doc_id,
                "document_name": doc_name,
                "page_number": page_num,
                "chunk_type": "child",
                "token_count": len(part1.split()),
                "tenant_id": "default",
            },
        )
    )
    if n_children > 1:
        children.append(
            Document(
                page_content=part2,
                metadata={
                    "chunk_id": f"child_{doc_id}_1",
                    "parent_id": parent_id,
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "chunk_type": "child",
                    "token_count": len(part2.split()),
                    "tenant_id": "default",
                },
            )
        )
    return children
