"""
SentinelRAG FastAPI Application
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.config import get_settings
from app.retrieval.qdrant import QdrantService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise services on startup."""
    logger.info("Starting SentinelRAG v%s (%s)", settings.app_version, settings.environment)

    # Ensure Qdrant collections exist
    try:
        qdrant = QdrantService()
        await qdrant.ensure_collections()
        logger.info("Qdrant collections verified")
    except Exception as exc:
        logger.warning("Qdrant init failed (will retry on first request): %s", exc)

    # Pre-compile LangGraph workflow
    try:
        from app.graph.workflow import get_workflow
        get_workflow()
        logger.info("LangGraph workflow compiled")
    except Exception as exc:
        logger.error("Failed to compile LangGraph workflow: %s", exc)

    yield

    logger.info("SentinelRAG shutting down")


app = FastAPI(
    title="SentinelRAG",
    description="Advanced Multi-Agent Document Intelligence & Evidence-Grounded RAG Platform",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Add request ID and timing to all responses."""
    import uuid
    request_id = str(uuid.uuid4())[:8]
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.0f}"
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured request logging. Never logs sensitive headers."""
    logger.info(
        "REQUEST %s %s",
        request.method,
        request.url.path,
    )
    response = await call_next(request)
    logger.info(
        "RESPONSE %s %s → %d",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "SentinelRAG",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
