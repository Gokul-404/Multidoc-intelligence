"""Redis client singleton with seamless in-memory fallback for SentinelRAG."""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)


class InMemoryRedis:
    """High-reliability in-memory Redis substitute when Redis daemon is offline with disk persistence."""

    def __init__(self) -> None:
        import os
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._storage_path = os.path.join(backend_dir, "data", "redis_storage.json")
        self._data: dict[str, Any] = {}
        self._sets: dict[str, set[str]] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._expires: dict[str, float] = {}
        self._load_from_disk()

    def _save_to_disk(self) -> None:
        try:
            import json, os
            os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "data": self._data,
                    "sets": {k: list(v) for k, v in self._sets.items()},
                    "hashes": self._hashes,
                }, f)
        except Exception as e:
            logger.debug("Failed to persist redis state: %s", e)

    def _load_from_disk(self) -> None:
        try:
            import json, os
            if os.path.exists(self._storage_path):
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    self._data = content.get("data", {})
                    self._sets = {k: set(v) for k, v in content.get("sets", {}).items()}
                    self._hashes = content.get("hashes", {})
        except Exception as e:
            logger.debug("Failed to load redis state: %s", e)

    def _is_expired(self, key: str) -> bool:
        if key in self._expires and time.time() > self._expires[key]:
            self._data.pop(key, None)
            self._sets.pop(key, None)
            self._hashes.pop(key, None)
            self._expires.pop(key, None)
            return True
        return False

    async def get(self, key: str) -> Optional[str]:
        if self._is_expired(key):
            return None
        return self._data.get(key)

    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        self._data[key] = str(value)
        if ex:
            self._expires[key] = time.time() + ex
        else:
            self._expires.pop(key, None)
        self._save_to_disk()
        return True

    async def setex(self, key: str, time_seconds: int, value: Any) -> bool:
        return await self.set(key, value, ex=time_seconds)

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._data or k in self._sets or k in self._hashes:
                self._data.pop(k, None)
                self._sets.pop(k, None)
                self._hashes.pop(k, None)
                self._expires.pop(k, None)
                count += 1
        if count > 0:
            self._save_to_disk()
        return count

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if (not self._is_expired(k)) and (k in self._data or k in self._sets or k in self._hashes))

    async def expire(self, key: str, seconds: int) -> bool:
        self._expires[key] = time.time() + seconds
        return True

    async def sadd(self, key: str, *members: str) -> int:
        self._is_expired(key)
        if key not in self._sets:
            self._sets[key] = set()
        prev_len = len(self._sets[key])
        self._sets[key].update(str(m) for m in members)
        diff = len(self._sets[key]) - prev_len
        if diff > 0:
            self._save_to_disk()
        return diff

    async def smembers(self, key: str) -> set[str]:
        if self._is_expired(key):
            return set()
        return set(self._sets.get(key, set()))

    async def srem(self, key: str, *members: str) -> int:
        if self._is_expired(key) or key not in self._sets:
            return 0
        count = 0
        for m in members:
            if m in self._sets[key]:
                self._sets[key].remove(m)
                count += 1
        if count > 0:
            self._save_to_disk()
        return count

    async def hset(self, key: str, mapping: Optional[dict] = None, **kwargs) -> int:
        self._is_expired(key)
        if key not in self._hashes:
            self._hashes[key] = {}
        merged = {**(mapping or {}), **kwargs}
        for k, v in merged.items():
            self._hashes[key][str(k)] = str(v)
        self._save_to_disk()
        return len(merged)

    async def hget(self, key: str, field: str) -> Optional[str]:
        if self._is_expired(key) or key not in self._hashes:
            return None
        return self._hashes[key].get(field)

    async def hgetall(self, key: str) -> dict[str, str]:
        if self._is_expired(key) or key not in self._hashes:
            return {}
        return dict(self._hashes[key])

    async def info(self, section: Optional[str] = None) -> dict[str, Any]:
        return {"used_memory_human": "in-memory-local"}

    async def ping(self) -> bool:
        return True


class ResilientRedisClient:
    """Transparent proxy that uses real Redis when available and falls back to in-memory."""

    def __init__(self, real_client: aioredis.Redis) -> None:
        self._real = real_client
        self._local = InMemoryRedis()
        self._is_offline = False

    def __getattr__(self, name: str):
        real_attr = getattr(self._real, name, None)
        local_attr = getattr(self._local, name, None)

        if not callable(real_attr):
            return getattr(self._local, name, None)

        async def _wrapper(*args, **kwargs):
            if not self._is_offline:
                try:
                    return await real_attr(*args, **kwargs)
                except Exception as exc:
                    logger.debug("Redis operation '%s' failed, using in-memory store: %s", name, exc)
                    self._is_offline = True
            
            if local_attr:
                return await local_attr(*args, **kwargs)
            logger.warning("Unimplemented in-memory redis method '%s'", name)
            return None

        return _wrapper


_GLOBAL_RESILIENT_REDIS: Optional[ResilientRedisClient] = None


@lru_cache(maxsize=1)
def get_redis_client() -> ResilientRedisClient:
    global _GLOBAL_RESILIENT_REDIS
    if _GLOBAL_RESILIENT_REDIS is None:
        settings = get_settings()
        real_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        _GLOBAL_RESILIENT_REDIS = ResilientRedisClient(real_client)
        logger.info("Resilient Redis proxy initialised")
    return _GLOBAL_RESILIENT_REDIS


async def check_redis_health() -> bool:
    """Return True if Redis or in-memory fallback is operational."""
    try:
        client = get_redis_client()
        await client.ping()
        return True
    except Exception:
        return False
