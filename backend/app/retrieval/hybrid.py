"""
Hybrid retrieval with Reciprocal Rank Fusion for SentinelRAG.
Combines Qdrant dense search + BM25 sparse search.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.retrieval.bm25 import BM25Service, RetrievedChunk
from app.retrieval.qdrant import QdrantService

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class HybridResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    dense_count: int = 0
    bm25_count: int = 0
    unique_count: int = 0
    dense_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    weights: Optional[list[float]] = None,
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.
    score(d) = sum_i [ weight_i / (k + rank_i(d)) ]
    Returns merged list sorted by descending RRF score.
    """
    if not ranked_lists:
        return []

    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id or chunk.document.metadata.get("chunk_id", id(chunk))
            rrf_score = weight / (k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf_score
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    # Sort by descending RRF score
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    result = []
    for cid in sorted_ids:
        chunk = chunk_map[cid]
        # Store the RRF score in the chunk
        result.append(RetrievedChunk(
            document=chunk.document,
            score=scores[cid],
            chunk_id=cid,
            retrieval_method="rrf",
        ))
    return result


class HybridRetriever:
    """
    Parallel dense + BM25 retrieval with Reciprocal Rank Fusion.
    Scores are normalised before fusion.
    """

    def __init__(
        self,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> None:
        self._qdrant = QdrantService()
        self._bm25 = BM25Service()
        self._dense_weight = dense_weight or settings.dense_weight
        self._bm25_weight = bm25_weight or settings.bm25_weight

    def _normalize_scores(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Min-max normalise scores to [0, 1]."""
        if not chunks:
            return chunks
        scores = [c.score for c in chunks]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s != min_s else 1.0
        return [
            RetrievedChunk(
                document=c.document,
                score=(c.score - min_s) / rng,
                chunk_id=c.chunk_id,
            )
            for c in chunks
        ]

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        document_ids: Optional[list[str]] = None,
        dense_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        hybrid_k: Optional[int] = None,
    ) -> HybridResult:
        """
        Run dense and BM25 retrieval in parallel, then apply RRF.
        Returns deduplicated, fused candidate set.
        """
        import time

        dk = dense_k or settings.dense_top_k
        bk = bm25_k or settings.bm25_top_k
        hk = hybrid_k or settings.hybrid_top_k

        t0 = time.monotonic()
        dense_task = asyncio.create_task(
            self._qdrant.search(query, tenant_id, document_ids, k=dk)
        )
        bm25_task = asyncio.create_task(
            self._bm25.search(query, tenant_id, document_ids, k=bk)
        )

        dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
        t1 = time.monotonic()

        dense_latency = (t1 - t0) * 1000
        bm25_latency = dense_latency  # ran concurrently

        # Normalise separately before fusion
        norm_dense = self._normalize_scores(dense_results)
        norm_bm25 = self._normalize_scores(bm25_results)

        # RRF fusion with configurable weights
        fused = reciprocal_rank_fusion(
            ranked_lists=[norm_dense, norm_bm25],
            weights=[self._dense_weight, self._bm25_weight],
            k=settings.rrf_k,
        )

        # Trim to hybrid_k
        fused = fused[:hk]

        logger.debug(
            "Hybrid retrieval: dense=%d bm25=%d unique=%d",
            len(dense_results), len(bm25_results), len(fused),
        )

        return HybridResult(
            chunks=fused,
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            unique_count=len(fused),
            dense_latency_ms=dense_latency,
            bm25_latency_ms=bm25_latency,
        )
