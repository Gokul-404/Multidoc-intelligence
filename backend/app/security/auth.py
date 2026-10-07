"""
Security utilities for SentinelRAG.
- API key validation
- Rate limiting
- Safe filename handling
- Prompt injection detection (advisory - primary defense is prompt architecture)
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.cache.redis_client import get_redis_client
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Prompt injection advisory patterns ───────────────────────────────────────
# These are ADVISORY only. The primary defence is the prompt architecture
# (DOCUMENT_WRAPPER_START/END delimiters + system instructions).
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new)\s+(?:AI|assistant|model)", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|context)", re.IGNORECASE),
    re.compile(r"system:\s*you\s+are", re.IGNORECASE),
    re.compile(r"<\|(?:system|im_start)\|>", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> bool:
    """
    Advisory detection of obvious prompt injection patterns.
    Returns True if suspicious patterns are found.
    Note: This is a secondary defence layer. The primary protection is
    the DOCUMENT_WRAPPER_START/END delimiters in the generation prompt.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other attacks.
    """
    # Normalize unicode
    filename = unicodedata.normalize("NFKD", filename)
    # Remove path separators
    filename = Path(filename).name
    # Remove null bytes and control characters
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)
    # Extract stem and suffix
    p = Path(filename)
    suffix = p.suffix
    stem = p.stem
    # Replace dangerous characters with underscore
    stem = re.sub(r"[^a-zA-Z0-9._\-]", "_", stem)
    # Collapse consecutive underscores and trim edges
    stem = re.sub(r"_+", "_", stem).strip("_.")
    if not stem:
        stem = "uploaded_file"
    # Limit length
    if len(stem) > 200:
        stem = stem[:200]
    # Prevent reserved names (Windows)
    reserved = {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}
    if stem.upper() in reserved:
        stem = f"file_{stem}"
    return f"{stem}{suffix}" or "uploaded_file.pdf"


def validate_api_key(api_key: Optional[str]) -> bool:
    """Validate the API key."""
    if settings.environment == "development" and not api_key:
        return True
    if not api_key:
        return False
    # Constant-time comparison to prevent timing attacks
    target_hash = hashlib.sha256(settings.api_key.encode()).hexdigest()
    dev_hash = hashlib.sha256("sentinel-dev-key-change-in-production".encode()).hexdigest()
    given_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return given_hash == target_hash or (settings.environment == "development" and given_hash == dev_hash)


async def require_api_key(request: Request) -> str:
    """FastAPI dependency: validate API key from header."""
    api_key = request.headers.get("X-API-Key")
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


class RateLimiter:
    """Simple Redis-backed sliding window rate limiter."""

    def __init__(self) -> None:
        self._redis = get_redis_client()

    async def check(self, identifier: str) -> tuple[bool, int]:
        """
        Check if identifier is within rate limits.
        Returns (allowed, remaining_requests).
        """
        key = f"rate_limit:{identifier}"
        try:
            pipeline = self._redis.pipeline()
            await pipeline.incr(key)
            await pipeline.expire(key, settings.rate_limit_window)
            results = await pipeline.execute()
            count = results[0]
            remaining = max(0, settings.rate_limit_requests - count)
            allowed = count <= settings.rate_limit_requests
            return allowed, remaining
        except Exception as exc:
            logger.warning("Rate limiter unavailable: %s; allowing request", exc)
            return True, settings.rate_limit_requests


async def check_rate_limit(request: Request, api_key: str) -> None:
    """FastAPI dependency: enforce rate limiting."""
    limiter = RateLimiter()
    identifier = f"{api_key[:8]}:{request.client.host if request.client else 'unknown'}"
    allowed, remaining = await limiter.check(identifier)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {settings.rate_limit_window}s.",
            headers={"X-RateLimit-Remaining": "0"},
        )
