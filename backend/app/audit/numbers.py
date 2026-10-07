"""
Numerical claim verification for SentinelRAG.
Handles different number formats and unit conversions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Number parsing ────────────────────────────────────────────────────────────

_MULTIPLIERS = {
    "thousand": 1_000,
    "k": 1_000,
    "million": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "t": 1_000_000_000_000,
    # Indian number system
    "lakh": 100_000,
    "crore": 10_000_000,
}

_CURRENCY_SYMBOLS = {"$", "€", "£", "₹", "¥", "₩", "₣"}

_NUMBER_RE = re.compile(
    r"""
    (?P<currency>[€£$₹¥₩])?       # Optional leading currency
    \s*
    (?P<sign>[+-])?                # Optional sign
    (?P<integer>[\d,]+)            # Integer part (may have commas)
    (?:\.(?P<decimal>\d+))?        # Optional decimal
    \s*
    (?P<multiplier>
        trillion|billion|million|thousand|crore|lakh|
        [tTbBmMkK](?!\w)           # single-letter multipliers (not followed by word char)
    )?
    \s*
    (?P<pct>%)?                    # Optional percentage
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass
class ParsedNumber:
    raw: str
    value: float
    is_percentage: bool = False
    currency: Optional[str] = None


def parse_number(text: str) -> Optional[ParsedNumber]:
    """
    Parse a number string into a normalised float value.
    Supports: 10M, $10M, 10 million, 1,000,000, 10%, ₹10 crore, etc.
    """
    text = text.strip()
    m = _NUMBER_RE.match(text)
    if not m:
        return None

    # Integer + decimal
    int_part = m.group("integer").replace(",", "")
    dec_part = m.group("decimal") or "0"
    try:
        value = float(f"{int_part}.{dec_part}")
    except ValueError:
        return None

    # Sign
    if m.group("sign") == "-":
        value = -value

    # Multiplier
    mult_str = (m.group("multiplier") or "").lower()
    multiplier = _MULTIPLIERS.get(mult_str, 1)
    value *= multiplier

    # Percentage normalisation – keep as percentage value (not fraction)
    is_pct = bool(m.group("pct"))

    currency = m.group("currency")

    return ParsedNumber(raw=text, value=value, is_percentage=is_pct, currency=currency)


def numbers_agree(
    generated: str,
    source: str,
    tolerance: float = 0.01,
) -> bool:
    """
    Check if two number strings represent the same value within tolerance.
    Handles different formats: "10M" == "10 million" == "$10,000,000"
    """
    gen = parse_number(generated)
    src = parse_number(source)
    if gen is None or src is None:
        return False
    if gen.is_percentage != src.is_percentage:
        return False  # Can't compare percentage to absolute
    if gen.value == 0 and src.value == 0:
        return True
    rel_diff = abs(gen.value - src.value) / max(abs(src.value), 1e-9)
    return rel_diff <= tolerance


# ── Claim number extraction ───────────────────────────────────────────────────

_NUMBER_CLAIM_RE = re.compile(
    r"""
    (?:[€£$₹¥]?\s*)?
    (?:[+-]?\d[\d,]*(?:\.\d+)?)\s*
    (?:trillion|billion|million|thousand|crore|lakh|[tTbBmMkK](?!\w))?
    \s*%?
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_numbers_from_text(text: str) -> list[str]:
    """Extract all numeric expressions from a text string."""
    matches = _NUMBER_CLAIM_RE.findall(text)
    return [m.strip() for m in matches if m.strip() and any(c.isdigit() for c in m)]


@dataclass
class NumericalVerificationResult:
    claim_text: str
    numbers_found: list[str]
    verified: bool
    unverified_numbers: list[str]
    failure_reason: Optional[str] = None


def verify_numerical_claims(
    generated_text: str,
    evidence_texts: list[str],
) -> NumericalVerificationResult:
    """
    Verify that all numbers in the generated text appear in the evidence.
    Returns a result indicating which numbers (if any) are unsupported.
    """
    gen_numbers = extract_numbers_from_text(generated_text)
    if not gen_numbers:
        return NumericalVerificationResult(
            claim_text=generated_text,
            numbers_found=[],
            verified=True,
            unverified_numbers=[],
        )

    all_evidence_numbers: list[str] = []
    for ev in evidence_texts:
        all_evidence_numbers.extend(extract_numbers_from_text(ev))

    unverified: list[str] = []
    for gen_num in gen_numbers:
        matched = any(numbers_agree(gen_num, ev_num) for ev_num in all_evidence_numbers)
        if not matched:
            # Skip years, counts, and very small standalone integers < 4 digits
            parsed = parse_number(gen_num)
            if parsed and abs(parsed.value) < 10000 and not parsed.is_percentage:
                continue  # Treat small integers as low-risk
            unverified.append(gen_num)

    verified = len(unverified) == 0
    failure_reason = (
        f"Numbers not found in evidence: {unverified}" if unverified else None
    )

    return NumericalVerificationResult(
        claim_text=generated_text,
        numbers_found=gen_numbers,
        verified=verified,
        unverified_numbers=unverified,
        failure_reason=failure_reason,
    )
