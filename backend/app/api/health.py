"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.cache.redis_client import check_redis_health, get_redis_client
from app.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    services: dict[str, str] = {}

    # Redis
    services["redis"] = "healthy" if await check_redis_health() else "unhealthy"

    # Qdrant
    try:
        from app.retrieval.qdrant import QdrantService
        qdrant = QdrantService()
        client = await qdrant._get_client()
        await client.get_collections()
        services["qdrant"] = "healthy"
    except Exception as exc:
        services["qdrant"] = f"unhealthy: {exc}"

    # Google API
    services["google_ai"] = "configured" if settings.google_api_key else "not_configured"

    overall = "healthy" if all(v == "healthy" or v == "configured" for v in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        services=services,
    )


@router.get("/metrics")
async def metrics():
    """Basic Prometheus-compatible metrics."""
    redis = get_redis_client()
    try:
        info = await redis.info()
        redis_memory = info.get("used_memory_human", "unknown")
    except Exception:
        redis_memory = "unavailable"

    lines = [
        "# HELP sentinelrag_up Whether SentinelRAG is running",
        "# TYPE sentinelrag_up gauge",
        "sentinelrag_up 1",
        f"# HELP sentinelrag_redis_memory Redis memory usage",
        f"# sentinelrag_redis_memory {redis_memory}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))
