"""
Intelligent chunking pipeline for SentinelRAG.
Converts a DocumentLayout into LangChain Documents with parent-child relationships.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.layout import DocumentLayout, LayoutBlock
from app.langchain_components.splitters import (
    build_parent_child_chunks,
    get_recursive_splitter,
    get_structure_aware_splitter,
)
from app.models.schemas import DocumentType

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ChunkingResult:
    parent_docs: list[Document] = field(default_factory=list)
    child_docs: list[Document] = field(default_factory=list)
    total_chunks: int = 0
    total_parents: int = 0


def _block_to_document(
    block: LayoutBlock,
    document_id: str,
    filename: str,
    tenant_id: str,
    document_type: DocumentType,
    language: str,
    author: Optional[str],
    creation_date: Optional[str],
) -> Optional[Document]:
    """Convert a LayoutBlock to a LangChain Document."""
    text = block.text.strip()
    if len(text) < settings.min_chunk_length:
        return None

    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "filename": filename,
            "tenant_id": tenant_id,
            "page": block.page,
            "section": block.section,
            "chunk_type": block.block_type,
            "document_type": document_type.value,
            "language": language,
            "author": author,
            "creation_date": creation_date,
        },
    )


def chunk_document(
    layout: DocumentLayout,
    document_id: str,
    filename: str,
    tenant_id: str,
    document_type: DocumentType = DocumentType.GENERAL,
    language: str = "en",
    author: Optional[str] = None,
    creation_date: Optional[str] = None,
) -> ChunkingResult:
    """
    Main chunking pipeline:
    1. Convert layout blocks to Documents
    2. Apply structure-aware splitting
    3. Build parent-child pairs
    4. Assign canonical chunk IDs
    """
    # Step 1: Convert layout blocks to Documents
    raw_docs: list[Document] = []
    for page_layout in layout.pages:
        for block in page_layout.blocks:
            doc = _block_to_document(
                block=block,
                document_id=document_id,
                filename=filename,
                tenant_id=tenant_id,
                document_type=document_type,
                language=language,
                author=author,
                creation_date=creation_date,
            )
            if doc:
                raw_docs.append(doc)

    if not raw_docs:
        logger.warning("No content blocks found for document %s", document_id)
        return ChunkingResult()

    # Step 2: Apply structure-aware splitting for oversized blocks
    splitter = get_structure_aware_splitter()
    split_docs: list[Document] = []
    for doc in raw_docs:
        if len(doc.page_content) > settings.chunk_size * 3:
            # Only split very long blocks; tables stay intact
            if doc.metadata.get("chunk_type") == "table":
                split_docs.append(doc)  # Never split tables
            else:
                sub_docs = splitter.split_documents([doc])
                # Propagate metadata to sub-docs
                for sd in sub_docs:
                    sd.metadata.update(doc.metadata)
                split_docs.extend(sub_docs)
        else:
            split_docs.append(doc)

    # Step 3: Build parent-child pairs
    parents, children = build_parent_child_chunks(split_docs)

    # Step 4: Assign final canonical chunk IDs
    for i, child in enumerate(children):
        parent_id = child.metadata.get("parent_id", f"{document_id}_p0")
        child.metadata["chunk_id"] = f"{parent_id}_c{i}"

    for i, parent in enumerate(parents):
        parent.metadata["chunk_id"] = f"{document_id}_p{i}"

    logger.info(
        "Chunked document %s: %d source blocks → %d parents, %d children",
        document_id,
        len(split_docs),
        len(parents),
        len(children),
    )

    return ChunkingResult(
        parent_docs=parents,
        child_docs=children,
        total_chunks=len(children),
        total_parents=len(parents),
    )
