"""Redis client singleton for SentinelRAG."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    settings = get_settings()
    client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    logger.info("Redis client initialised at %s", settings.redis_url)
    return client


async def check_redis_health() -> bool:
    """Return True if Redis is reachable."""
    try:
        client = get_redis_client()
        await client.ping()
        return True
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return False
