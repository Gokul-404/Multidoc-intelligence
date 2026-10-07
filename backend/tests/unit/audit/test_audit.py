"""
Unit tests for the SentinelRAG Evidence Audit subsystem.
Tests numerical verification, unit conversion, citation auditing, and confidence calculation.
"""
import pytest
from app.audit.numbers import (
    parse_number,
    numbers_agree,
    extract_numbers_from_text,
    verify_numerical_claims,
)
from app.audit.auditor import calculate_confidence_score


def test_parse_number_formats():
    p1 = parse_number("$15.4 million")
    assert p1 is not None
    assert p1.value == 15_400_000.0
    assert p1.currency == "$"

    p2 = parse_number("₹10 crore")
    assert p2 is not None
    assert p2.value == 100_000_000.0
    assert p2.currency == "₹"

    p3 = parse_number("24.5%")
    assert p3 is not None
    assert p3.value == 24.5
    assert p3.is_percentage is True

    p4 = parse_number("1,250,000")
    assert p4 is not None
    assert p4.value == 1_250_000.0

    p5 = parse_number("500k")
    assert p5 is not None
    assert p5.value == 500_000.0


def test_numbers_agree():
    # Matching across format differences
    assert numbers_agree("$10M", "$10 million") is True
    assert numbers_agree("15.4M", "15,400,000") is True
    assert numbers_agree("25%", "25.0%") is True
    
    # Non-matching
    assert numbers_agree("$10M", "$12M") is False
    assert numbers_agree("25%", "28%") is False


def test_verify_numerical_claims_success():
    gen_text = "The company recorded $15.4 million in quarterly revenue, an increase of 18%."
    evidence = [
        "In Q3, total revenues reached $15.4M, representing an 18% year-over-year growth rate."
    ]
    result = verify_numerical_claims(gen_text, evidence)
    assert result.verified is True
    assert len(result.unverified_numbers) == 0


def test_verify_numerical_claims_failure():
    gen_text = "The company generated $99.9 million in revenue."
    evidence = [
        "Total revenue reached $15.4M for the quarter."
    ]
    result = verify_numerical_claims(gen_text, evidence)
    assert result.verified is False
    assert any("99.9" in n for n in result.unverified_numbers)


def test_calculate_confidence_score_deterministic():
    # Perfect audit
    score_perfect = calculate_confidence_score(
        claim_support_rate=1.0,
        citation_validity_rate=1.0,
        numerical_accuracy_rate=1.0,
        contradiction_penalty=0.0,
        mean_retrieval_score=0.9,
    )
    assert score_perfect >= 90.0

    # With penalties
    score_penalized = calculate_confidence_score(
        claim_support_rate=0.5,
        citation_validity_rate=0.5,
        numerical_accuracy_rate=0.0,
        contradiction_penalty=0.3,
        mean_retrieval_score=0.4,
    )
    assert score_penalized < score_perfect
    assert 0.0 <= score_penalized <= 100.0
