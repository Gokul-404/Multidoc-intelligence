"""
Deterministic Generation and Grounding evaluation metrics for SentinelRAG.
Implements: Faithfulness, Citation Precision & Recall, Refusal Accuracy, and Compression Ratio.
"""
from __future__ import annotations

from typing import List, Optional, Set
from dataclasses import dataclass


@dataclass
class GenerationEvalResult:
    faithfulness: float
    citation_precision: float
    citation_recall: float
    refusal_accuracy: float
    token_compression_ratio: float
    supported_claims: int
    total_claims: int
    valid_citations: int
    total_citations: int


def compute_faithfulness(supported_claims: int, total_claims: int) -> float:
    """Faithfulness = supported_claims / total_claims."""
    if total_claims == 0:
        return 1.0
    return min(1.0, max(0.0, supported_claims / total_claims))


def compute_citation_precision(valid_citations: int, total_citations: int) -> float:
    """Citation Precision = valid_citations / total_citations."""
    if total_citations == 0:
        return 1.0  # Vacuous if no claims required citations
    return min(1.0, max(0.0, valid_citations / total_citations))


def compute_citation_recall(cited_chunk_ids: Set[str], ground_truth_chunk_ids: Set[str]) -> float:
    """Citation Recall / Completeness = cited ground truth chunks / total ground truth chunks."""
    if not ground_truth_chunk_ids:
        return 1.0
    hits = len(cited_chunk_ids.intersection(ground_truth_chunk_ids))
    return min(1.0, max(0.0, hits / len(ground_truth_chunk_ids)))


def compute_refusal_accuracy(
    actual_refusals: List[bool],
    expected_refusals: List[bool],
) -> float:
    """Accuracy of refusal decisions (both refusing when unanswerable and answering when answerable)."""
    if not expected_refusals:
        return 1.0
    correct = sum(1 for a, e in zip(actual_refusals, expected_refusals) if a == e)
    return correct / len(expected_refusals)


def compute_compression_ratio(compressed_tokens: int, uncompressed_tokens: int) -> float:
    """Token compression ratio = 1.0 - (compressed / uncompressed). Higher means more reduction."""
    if uncompressed_tokens <= 0:
        return 0.0
    ratio = 1.0 - (compressed_tokens / uncompressed_tokens)
    return round(max(0.0, ratio), 4)
