"""
Response caching for SentinelRAG.
Cache key includes: question + document_ids + retrieval_config hash.
Cache is invalidated when relevant documents change.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from app.cache.redis_client import get_redis_client
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _make_cache_key(
    question: str,
    tenant_id: str,
    document_ids: Optional[list[str]],
    retrieval_config_dict: Optional[dict],
) -> str:
    """Deterministic cache key that includes all query-affecting parameters."""
    doc_ids_str = json.dumps(sorted(document_ids or []))
    config_str = json.dumps(retrieval_config_dict or {}, sort_keys=True)
    raw = f"{tenant_id}:{question}:{doc_ids_str}:{config_str}"
    return f"cache:query:{hashlib.sha256(raw.encode()).hexdigest()}"


class QueryCache:
    """Redis-backed response cache with TTL."""

    def __init__(self) -> None:
        self._redis = get_redis_client()

    async def get(
        self,
        question: str,
        tenant_id: str,
        document_ids: Optional[list[str]] = None,
        retrieval_config_dict: Optional[dict] = None,
    ) -> Optional[dict]:
        """Return cached response dict or None."""
        key = _make_cache_key(question, tenant_id, document_ids, retrieval_config_dict)
        try:
            raw = await self._redis.get(key)
            if raw:
                logger.debug("Cache HIT: %s", key[:40])
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Cache get failed (degrading gracefully): %s", exc)
        return None

    async def set(
        self,
        question: str,
        tenant_id: str,
        response_dict: dict,
        document_ids: Optional[list[str]] = None,
        retrieval_config_dict: Optional[dict] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache a response. Gracefully degrades if Redis is unavailable."""
        key = _make_cache_key(question, tenant_id, document_ids, retrieval_config_dict)
        try:
            await self._redis.setex(
                key,
                ttl or settings.redis_cache_ttl,
                json.dumps(response_dict, default=str),
            )
            # Track key under tenant for bulk invalidation
            await self._redis.sadd(f"cache:tenant_keys:{tenant_id}", key)
            logger.debug("Cache SET: %s", key[:40])
        except Exception as exc:
            logger.warning("Cache set failed (degrading gracefully): %s", exc)

    async def invalidate_for_document(self, document_id: str, tenant_id: str) -> None:
        """Invalidate all cached responses for a tenant (conservative on delete)."""
        try:
            keys = await self._redis.smembers(f"cache:tenant_keys:{tenant_id}")
            if keys:
                await self._redis.delete(*keys)
                await self._redis.delete(f"cache:tenant_keys:{tenant_id}")
            logger.info("Invalidated %d cache entries for tenant %s", len(keys or []), tenant_id)
        except Exception as exc:
            logger.warning("Cache invalidation failed: %s", exc)
