"""
Deterministic Information Retrieval evaluation metrics for SentinelRAG.
Implements: Recall@K, Precision@K, MRR@K, NDCG@K.
"""
from __future__ import annotations

import math
from typing import List, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Precision@K = (# relevant items in top K) / K"""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for cid in top_k if cid in relevant_ids)
    return relevant_retrieved / float(k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Recall@K = (# relevant items in top K) / (total relevant items)"""
    if not relevant_ids:
        return 1.0  # Vacuously true if nothing was relevant
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for cid in top_k if cid in relevant_ids)
    return relevant_retrieved / float(len(relevant_ids))


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Mean Reciprocal Rank @ K: 1 / rank of first relevant item in top K, or 0."""
    top_k = retrieved_ids[:k]
    for rank, cid in enumerate(top_k, start=1):
        if cid in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K with binary relevance.
    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    IDCG@K = ideal DCG with all relevant items up to K at top ranks.
    """
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, cid in enumerate(top_k, start=1):
        rel = 1.0 if cid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 1.0 if not relevant_ids else 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_benchmark(
    queries_results: List[dict],
    k_values: List[int] = [3, 5, 10],
) -> dict:
    """
    Compute aggregate retrieval metrics over an evaluation dataset.
    queries_results should be a list of dicts:
    {"query_id": str, "retrieved_ids": List[str], "relevant_ids": Set[str]}
    """
    metrics = {}
    n = len(queries_results)
    if n == 0:
        return metrics

    for k in k_values:
        total_p = sum(precision_at_k(q["retrieved_ids"], q["relevant_ids"], k) for q in queries_results)
        total_r = sum(recall_at_k(q["retrieved_ids"], q["relevant_ids"], k) for q in queries_results)
        total_mrr = sum(mrr_at_k(q["retrieved_ids"], q["relevant_ids"], k) for q in queries_results)
        total_ndcg = sum(ndcg_at_k(q["retrieved_ids"], q["relevant_ids"], k) for q in queries_results)

        metrics[f"precision@{k}"] = round(total_p / n, 4)
        metrics[f"recall@{k}"] = round(total_r / n, 4)
        metrics[f"mrr@{k}"] = round(total_mrr / n, 4)
        metrics[f"ndcg@{k}"] = round(total_ndcg / n, 4)

    return metrics
