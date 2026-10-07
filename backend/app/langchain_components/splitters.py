"""
LangChain text splitters for SentinelRAG.
Implements recursive, structure-aware, and parent-child chunking strategies.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


# ── Child / Retrieval Chunks ──────────────────────────────────────────────────

def get_recursive_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Standard recursive splitter used for child chunks."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        keep_separator=True,
    )


def get_parent_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Larger splitter used for parent context chunks."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.parent_chunk_size,
        chunk_overlap=chunk_overlap or settings.parent_chunk_overlap,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
        keep_separator=True,
    )


# ── Structure-aware splitter ───────────────────────────────────────────────────

SECTION_SEPARATORS = [
    r"(?:\n|^)#{1,6}\s",          # Markdown headings
    r"(?:\n|^)[A-Z][A-Z\s]{4,}(?:\n|$)", # ALL-CAPS headings
    "\n\n\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def get_structure_aware_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Structure-aware splitter that tries to honour section boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        length_function=len,
        separators=SECTION_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
    )


# ── Parent-Child Construction ─────────────────────────────────────────────────

def build_parent_child_chunks(
    docs: list[Document],
    parent_chunk_size: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
) -> tuple[list[Document], list[Document]]:
    """
    Split each document into parent chunks (large) and child chunks (small).

    Returns:
        parents: list of parent Documents with metadata['parent_id'] and ['chunk_id']
        children: list of child Documents with metadata['parent_id'] referencing their parent
    """
    parent_splitter = get_parent_splitter(chunk_size=parent_chunk_size)
    child_splitter = get_recursive_splitter(chunk_size=child_chunk_size)

    parents: list[Document] = []
    children: list[Document] = []

    for doc in docs:
        doc_id = doc.metadata.get("document_id", "unknown")
        filename = doc.metadata.get("filename", "unknown")

        # Create parents
        parent_docs = parent_splitter.split_documents([doc])
        for p_idx, parent_doc in enumerate(parent_docs):
            parent_id = f"{doc_id}_p{p_idx}"
            parent_doc.metadata.update(
                {
                    "parent_id": parent_id,
                    "chunk_id": parent_id,
                    "chunk_type": "parent",
                    "document_id": doc_id,
                    "filename": filename,
                }
            )
            parents.append(parent_doc)

            # Create children from this parent
            child_docs = child_splitter.split_documents([parent_doc])
            for c_idx, child_doc in enumerate(child_docs):
                chunk_id = f"{parent_id}_c{c_idx}"
                child_doc.metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        "document_id": doc_id,
                        "filename": filename,
                    }
                )
                if len(child_doc.page_content.strip()) >= settings.min_chunk_length:
                    children.append(child_doc)

    logger.info(
        "Built %d parents and %d children from %d source documents",
        len(parents),
        len(children),
        len(docs),
    )
    return parents, children
