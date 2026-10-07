"""
Claim-level evidence auditor for SentinelRAG.
Breaks the generated answer into claims and verifies each against retrieved evidence.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from app.langchain_components.models import get_model_service
from app.langchain_components.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_VERIFICATION_PROMPT
from app.models.schemas import ClaimVerification

logger = logging.getLogger(__name__)


@dataclass
class ClaimsAuditResult:
    claims: list[ClaimVerification] = field(default_factory=list)
    total: int = 0
    supported: int = 0
    unsupported: int = 0
    all_supported: bool = True


def _format_evidence(docs: list[Document]) -> str:
    """Format retrieved documents for the claim verification prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Evidence {i}] chunk_id={chunk_id} | {filename} p.{page}\n"
            f"<<<DOCUMENT_CONTENT_START>>>\n{doc.page_content}\n<<<DOCUMENT_CONTENT_END>>>"
        )
    return "\n\n".join(parts)


class ClaimsAuditor:
    """
    Extracts claims from generated answer and verifies each against evidence.
    Uses LangChain prompt templates and model abstraction.
    """

    def __init__(self) -> None:
        self._model = get_model_service()

    def _extract_claims(self, answer: str) -> list[str]:
        """Use LLM to extract verifiable claims from the answer."""
        prompt = CLAIM_EXTRACTION_PROMPT.format_messages(answer=answer)
        try:
            raw = self._model.generate(prompt)
            parsed = self._model.parse_json_response(raw)
            if isinstance(parsed, list):
                return [str(c) for c in parsed if str(c).strip()]
        except Exception as exc:
            logger.warning("Claim extraction failed: %s", exc)
        return []

    def _verify_claim(
        self,
        claim: str,
        evidence_docs: list[Document],
    ) -> ClaimVerification:
        """Verify a single claim against the evidence."""
        formatted = _format_evidence(evidence_docs)
        prompt = CLAIM_VERIFICATION_PROMPT.format_messages(
            claim=claim,
            formatted_evidence=formatted,
        )
        try:
            raw = self._model.generate(prompt)
            parsed = self._model.parse_json_response(raw)
            if parsed:
                return ClaimVerification(
                    claim=claim,
                    supported=bool(parsed.get("supported", False)),
                    supporting_chunk_ids=parsed.get("supporting_chunk_ids", []),
                    evidence_text=parsed.get("evidence_text"),
                    failure_reason=parsed.get("failure_reason"),
                )
        except Exception as exc:
            logger.warning("Claim verification failed for '%s': %s", claim[:60], exc)
        # Default to unsupported on error (conservative)
        return ClaimVerification(
            claim=claim,
            supported=False,
            failure_reason="Verification error; treating as unsupported.",
        )

    def audit(
        self,
        answer: str,
        evidence_docs: list[Document],
        skip_if_refused: bool = True,
    ) -> ClaimsAuditResult:
        """
        Full claim-level audit pipeline.
        Extracts claims, verifies each, returns structured result.
        """
        REFUSAL_INDICATORS = [
            "couldn't verify",
            "cannot verify",
            "I don't know",
            "no information",
            "not in the documents",
        ]

        # If the answer is a refusal, all claims are implicitly supported
        if skip_if_refused and any(ind.lower() in answer.lower() for ind in REFUSAL_INDICATORS):
            logger.debug("Answer is a refusal; skipping claim audit")
            return ClaimsAuditResult(all_supported=True)

        claims = self._extract_claims(answer)
        if not claims:
            logger.debug("No verifiable claims found in answer")
            return ClaimsAuditResult(all_supported=True)

        verifications: list[ClaimVerification] = []
        for claim in claims:
            result = self._verify_claim(claim, evidence_docs)
            verifications.append(result)

        supported = sum(1 for v in verifications if v.supported)
        unsupported = len(verifications) - supported

        return ClaimsAuditResult(
            claims=verifications,
            total=len(verifications),
            supported=supported,
            unsupported=unsupported,
            all_supported=unsupported == 0,
        )
