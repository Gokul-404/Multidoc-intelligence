"""
FlashRank reranker for SentinelRAG.
Cross-encoder reranking of hybrid retrieval candidates.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.retrieval.qdrant import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk]
    input_count: int
    output_count: int
    latency_ms: float


class Reranker:
    """
    FlashRank-based cross-encoder reranker.
    Scores each (query, chunk) pair and reorders by relevance.
    Falls back to original order if FlashRank is unavailable.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._ranker = None
        self._available = False
        self._init_ranker()

    def _init_ranker(self) -> None:
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self._model_name, cache_dir="/tmp/flashrank_cache")
            self._available = True
            logger.info("FlashRank reranker initialised: %s", self._model_name)
        except ImportError:
            logger.warning("FlashRank not installed; reranking will use retrieval scores")
        except Exception as exc:
            logger.warning("FlashRank init failed: %s; using fallback", exc)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Rerank chunks by cross-encoder relevance.
        Returns top_k chunks sorted by rerank score.
        """
        top_k = top_k or settings.rerank_top_k
        input_count = len(chunks)
        t0 = time.monotonic()

        if not chunks:
            return RerankResult(chunks=[], input_count=0, output_count=0, latency_ms=0.0)

        if self._available and self._ranker:
            try:
                from flashrank import RerankRequest

                passages = [
                    {"id": i, "text": c.document.page_content}
                    for i, c in enumerate(chunks)
                ]
                request = RerankRequest(query=query, passages=passages)
                results = self._ranker.rerank(request)

                # Build output preserving original chunk metadata
                reranked = []
                for result in results[:top_k]:
                    original_idx = result.get("id", 0)
                    original_chunk = chunks[original_idx]
                    reranked.append(
                        RetrievedChunk(
                            document=original_chunk.document,
                            score=float(result.get("score", original_chunk.score)),
                            chunk_id=original_chunk.chunk_id,
                        )
                    )

                latency_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "Reranked %d → %d chunks in %.0fms", input_count, len(reranked), latency_ms
                )
                return RerankResult(
                    chunks=reranked,
                    input_count=input_count,
                    output_count=len(reranked),
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                logger.error("Reranking failed: %s; falling back to retrieval scores", exc)

        # Fallback: sort by retrieval score and truncate
        fallback = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
        latency_ms = (time.monotonic() - t0) * 1000
        return RerankResult(
            chunks=fallback,
            input_count=input_count,
            output_count=len(fallback),
            latency_ms=latency_ms,
        )
