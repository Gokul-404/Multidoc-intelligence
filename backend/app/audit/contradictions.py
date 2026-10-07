"""
Contradiction detector for SentinelRAG.
Detects conflicting information across retrieved evidence chunks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

from app.langchain_components.models import get_model_service
from app.langchain_components.prompts import CONTRADICTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ContradictionResult:
    has_contradiction: bool
    details: Optional[str] = None


class ContradictionDetector:
    """Uses LLM to detect contradictory information across evidence chunks."""

    def __init__(self) -> None:
        self._model = get_model_service()

    def detect(
        self,
        question: str,
        evidence_docs: list[Document],
    ) -> ContradictionResult:
        """
        Check evidence chunks for contradictions relevant to the question.
        Only runs if we have at least 2 chunks from different documents.
        """
        if len(evidence_docs) < 2:
            return ContradictionResult(has_contradiction=False)

        # Only bother if chunks come from multiple documents
        doc_ids = {d.metadata.get("document_id", "") for d in evidence_docs}
        if len(doc_ids) < 2:
            return ContradictionResult(has_contradiction=False)

        # Format evidence
        parts = []
        for i, doc in enumerate(evidence_docs, 1):
            filename = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(
                f"[{i}] {filename} p.{page}:\n"
                f"<<<DOCUMENT_CONTENT_START>>>\n{doc.page_content}\n<<<DOCUMENT_CONTENT_END>>>"
            )
        formatted = "\n\n".join(parts)

        prompt = CONTRADICTION_PROMPT.format_messages(
            question=question,
            formatted_evidence=formatted,
        )

        try:
            raw = self._model.generate(prompt)
            parsed = self._model.parse_json_response(raw)
            if parsed:
                return ContradictionResult(
                    has_contradiction=bool(parsed.get("has_contradiction", False)),
                    details=parsed.get("contradiction_details"),
                )
        except Exception as exc:
            logger.warning("Contradiction detection failed: %s", exc)

        return ContradictionResult(has_contradiction=False)
