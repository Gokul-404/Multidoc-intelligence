"""
Unit tests for numerical claim verification.
"""
import pytest
from app.audit.numbers import (
    parse_number,
    numbers_agree,
    extract_numbers_from_text,
    verify_numerical_claims,
)


class TestParseNumber:

    def test_simple_integer(self):
        n = parse_number("42")
        assert n is not None
        assert n.value == 42.0

    def test_million_abbreviation(self):
        n = parse_number("10M")
        assert n is not None
        assert n.value == 10_000_000

    def test_million_full_word(self):
        n = parse_number("10 million")
        assert n is not None
        assert n.value == 10_000_000

    def test_billion(self):
        n = parse_number("1 billion")
        assert n is not None
        assert n.value == 1_000_000_000

    def test_currency_dollar(self):
        n = parse_number("$10M")
        assert n is not None
        assert n.value == 10_000_000
        assert n.currency == "$"

    def test_currency_rupee(self):
        n = parse_number("₹10 crore")
        assert n is not None
        assert n.value == 10 * 10_000_000  # 10 crore

    def test_percentage(self):
        n = parse_number("23%")
        assert n is not None
        assert n.is_percentage
        assert n.value == 23.0

    def test_comma_formatted(self):
        n = parse_number("1,000,000")
        assert n is not None
        assert n.value == 1_000_000

    def test_decimal(self):
        n = parse_number("10.5M")
        assert n is not None
        assert abs(n.value - 10_500_000) < 1

    def test_negative(self):
        n = parse_number("-10%")
        assert n is not None
        assert n.value == -10.0
        assert n.is_percentage

    def test_invalid_returns_none(self):
        n = parse_number("not a number")
        assert n is None


class TestNumbersAgree:

    def test_same_value_different_format(self):
        assert numbers_agree("10M", "10 million")
        assert numbers_agree("$10M", "10000000")
        assert numbers_agree("1 billion", "1000 million")

    def test_mismatch_detected(self):
        assert not numbers_agree("$20M", "$10M")
        assert not numbers_agree("10%", "20%")

    def test_tolerance_respected(self):
        # 10.0M vs 10.0M - exact match
        assert numbers_agree("10M", "10000000")
        # 10M vs 10.1M - within 1% tolerance
        assert numbers_agree("10M", "10.05M")

    def test_percentage_vs_absolute_mismatch(self):
        # 10% vs 10 (absolute) should not match
        assert not numbers_agree("10%", "10")


class TestExtractNumbers:

    def test_extracts_simple_numbers(self):
        numbers = extract_numbers_from_text("Revenue was $10M in Q3 2024.")
        assert any("10" in n for n in numbers)

    def test_handles_empty_text(self):
        numbers = extract_numbers_from_text("")
        assert numbers == []


class TestVerifyNumericalClaims:

    def test_supported_number_passes(self):
        generated = "Revenue was $10 million."
        evidence = ["Q4 revenue: $10M", "Year-end results show $10 million revenue."]
        result = verify_numerical_claims(generated, evidence)
        assert result.verified

    def test_unsupported_large_number_fails(self):
        generated = "Revenue was $20 million."
        evidence = ["Revenue was $10 million."]
        result = verify_numerical_claims(generated, evidence)
        assert not result.verified
        assert len(result.unverified_numbers) > 0

    def test_no_numbers_passes(self):
        generated = "The system uses a distributed architecture."
        evidence = ["The architecture is distributed across multiple nodes."]
        result = verify_numerical_claims(generated, evidence)
        assert result.verified

    def test_percentage_mismatch_fails(self):
        generated = "Latency improved by 45%."
        evidence = ["Latency improved by 23%."]
        result = verify_numerical_claims(generated, evidence)
        assert not result.verified
