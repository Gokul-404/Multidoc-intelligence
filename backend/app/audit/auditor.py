"""
Main Evidence Auditor for SentinelRAG.
Orchestrates: claims audit, citation audit, numerical verification,
contradiction detection, and system-level confidence scoring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

from app.audit.citations import CitationsAuditReport, audit_citations
from app.audit.claims import ClaimsAuditor, ClaimsAuditResult
from app.audit.contradictions import ContradictionDetector, ContradictionResult
from app.audit.numbers import verify_numerical_claims
from app.models.schemas import AuditResult, AuditStatus, Citation

logger = logging.getLogger(__name__)


def calculate_confidence_score(
    claim_support_rate: float = 1.0,
    citation_validity_rate: float = 1.0,
    numerical_accuracy_rate: float = 1.0,
    contradiction_penalty: float = 0.0,
    mean_retrieval_score: float = 0.8,
) -> float:
    """
    Compute a system-level confidence score [0, 100].
    Based on multi-signal grounding: claim support, citation validity,
    numerical verification, contradiction absence, and retrieval rank.
    Explicitly NOT LLM self-reported confidence.
    """
    score = 100.0
    score -= (1.0 - max(0.0, min(1.0, claim_support_rate))) * 35.0
    score -= (1.0 - max(0.0, min(1.0, citation_validity_rate))) * 25.0
    score -= (1.0 - max(0.0, min(1.0, numerical_accuracy_rate))) * 15.0
    score -= max(0.0, min(1.0, contradiction_penalty)) * 15.0
    score -= max(0.0, (0.5 - mean_retrieval_score)) * 20.0
    return max(0.0, min(100.0, score))


class EvidenceAuditor:
    """
    Master auditor that orchestrates all audit sub-systems and computes
    a system-derived confidence score based on multiple signals.
    """

    def __init__(self) -> None:
        self._claims_auditor = ClaimsAuditor()
        self._contradiction_detector = ContradictionDetector()

    def _compute_confidence(
        self,
        claims_result: ClaimsAuditResult,
        citations_report: CitationsAuditReport,
        numerical_verified: bool,
        has_contradiction: bool,
        reranker_scores: list[float],
    ) -> float:
        claim_rate = (claims_result.supported / claims_result.total) if claims_result.total > 0 else 1.0
        total_cit = citations_report.valid_count + citations_report.invalid_count
        cit_rate = (citations_report.valid_count / total_cit) if total_cit > 0 else 1.0
        num_rate = 1.0 if numerical_verified else 0.0
        contra_pen = 1.0 if has_contradiction else 0.0
        mean_ret = (sum(reranker_scores) / len(reranker_scores)) if reranker_scores else 0.5
        return calculate_confidence_score(
            claim_support_rate=claim_rate,
            citation_validity_rate=cit_rate,
            numerical_accuracy_rate=num_rate,
            contradiction_penalty=contra_pen,
            mean_retrieval_score=mean_ret,
        )

    def audit(
        self,
        answer: str,
        citations: list[Citation],
        evidence_docs: list[Document],
        question: str,
        max_page_by_doc: Optional[dict[str, int]] = None,
        reranker_scores: Optional[list[float]] = None,
    ) -> AuditResult:
        """
        Full evidence audit pipeline.
        Returns AuditResult with pass/fail status and detailed metrics.
        """
        max_page_by_doc = max_page_by_doc or {}
        reranker_scores = reranker_scores or []
        failure_reasons: list[str] = []

        # 1. Claim-level verification
        claims_result = self._claims_auditor.audit(answer, evidence_docs)
        if not claims_result.all_supported:
            for v in claims_result.claims:
                if not v.supported and v.failure_reason:
                    failure_reasons.append(f"Unsupported claim: {v.claim[:80]}...")

        # 2. Citation audit
        citations_report = audit_citations(citations, evidence_docs, max_page_by_doc)
        if not citations_report.all_valid:
            for r in citations_report.results:
                if not r.valid and r.failure_reason:
                    failure_reasons.append(f"Citation [{r.citation_index}]: {r.failure_reason}")

        # 3. Numerical verification
        evidence_texts = [d.page_content for d in evidence_docs]
        num_result = verify_numerical_claims(answer, evidence_texts)
        if not num_result.verified:
            failure_reasons.append(f"Numerical: {num_result.failure_reason}")

        # 4. Contradiction detection
        contradiction = self._contradiction_detector.detect(question, evidence_docs)
        if contradiction.has_contradiction:
            failure_reasons.append(f"Contradiction: {contradiction.details or 'Conflicting evidence detected'}")

        # 5. Compute confidence
        confidence = self._compute_confidence(
            claims_result=claims_result,
            citations_report=citations_report,
            numerical_verified=num_result.verified,
            has_contradiction=contradiction.has_contradiction,
            reranker_scores=reranker_scores,
        )

        # 6. Overall status
        critical_failure = (
            not claims_result.all_supported
            or not citations_report.all_valid
            or not num_result.verified
        )
        status = AuditStatus.FAIL if critical_failure else AuditStatus.PASS

        grounded = (
            claims_result.all_supported
            and citations_report.all_valid
            and num_result.verified
            and not contradiction.has_contradiction
        )

        logger.info(
            "Audit complete: status=%s confidence=%.0f%% claims=%d/%d citations=%d/%d numerical=%s contradiction=%s",
            status.value,
            confidence,
            claims_result.supported,
            claims_result.total,
            citations_report.valid_count,
            citations_report.valid_count + citations_report.invalid_count,
            num_result.verified,
            contradiction.has_contradiction,
        )

        return AuditResult(
            status=status,
            claims_total=claims_result.total,
            claims_supported=claims_result.supported,
            claims_unsupported=claims_result.unsupported,
            citations_valid=citations_report.valid_count,
            citations_invalid=citations_report.invalid_count,
            has_numerical_claims=bool(num_result.numbers_found),
            numerical_claims_verified=num_result.verified,
            has_contradictions=contradiction.has_contradiction,
            contradiction_details=contradiction.details,
            confidence_score=confidence,
            grounded=grounded,
            failure_reasons=failure_reasons,
        )
