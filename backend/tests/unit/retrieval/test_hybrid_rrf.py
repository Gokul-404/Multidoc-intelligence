"""
Unit tests for hybrid search and Reciprocal Rank Fusion (RRF).
"""
import pytest
from langchain_core.documents import Document

from app.retrieval.bm25 import RetrievedChunk
from app.retrieval.hybrid import reciprocal_rank_fusion


def make_chunk(cid: str, text: str, score: float = 1.0) -> RetrievedChunk:
    doc = Document(page_content=text, metadata={"chunk_id": cid, "document_id": "doc_1"})
    return RetrievedChunk(
        chunk_id=cid,
        document=doc,
        score=score,
        retrieval_method="dense",
    )


def test_rrf_empty_lists():
    res = reciprocal_rank_fusion([])
    assert res == []


def test_rrf_scoring_and_ranking():
    chunk_a = make_chunk("chunk_a", "Alpha text")
    chunk_b = make_chunk("chunk_b", "Beta text")
    chunk_c = make_chunk("chunk_c", "Gamma text")

    # List 1: [a, b]
    # List 2: [b, c]
    list1 = [chunk_a, chunk_b]
    list2 = [chunk_b, chunk_c]

    # chunk_b is ranked #2 in list 1 and #1 in list 2
    # chunk_a is ranked #1 in list 1 only
    # chunk_c is ranked #2 in list 2 only
    fused = reciprocal_rank_fusion([list1, list2], k=60)

    assert len(fused) == 3
    # chunk_b should have highest combined score because it appears in both lists:
    # score(b) = 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.0325
    # score(a) = 1/(60+1) = 0.016393
    # score(c) = 1/(60+2) = 0.016129
    assert fused[0].chunk_id == "chunk_b"
    assert fused[1].chunk_id == "chunk_a"
    assert fused[2].chunk_id == "chunk_c"
    assert fused[0].retrieval_method == "rrf"


def test_rrf_weights():
    chunk_a = make_chunk("chunk_a", "Alpha")
    chunk_b = make_chunk("chunk_b", "Beta")

    list1 = [chunk_a]  # weight 3.0
    list2 = [chunk_b]  # weight 1.0

    fused = reciprocal_rank_fusion([list1, list2], weights=[3.0, 1.0], k=60)
    assert fused[0].chunk_id == "chunk_a"
    assert fused[0].score > fused[1].score
