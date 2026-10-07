"""
SentinelRAG – Pydantic schemas
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class DocumentStatus(str, enum.Enum):
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    DETECTING_STRUCTURE = "detecting_structure"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DocumentType(str, enum.Enum):
    RESEARCH_PAPER = "research_paper"
    FINANCIAL_REPORT = "financial_report"
    LEGAL_DOCUMENT = "legal_document"
    TECHNICAL_DOCUMENT = "technical_document"
    RESUME = "resume"
    TEXTBOOK = "textbook"
    GENERAL = "general"


class QueryRoute(str, enum.Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    MULTI_DOCUMENT = "multi_document"
    SUMMARIZATION = "summarization"
    NUMERICAL = "numerical"
    UNSUPPORTED = "unsupported"


class AuditStatus(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


# ── Document Schemas ──────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    tenant_id: str
    status: DocumentStatus = DocumentStatus.UPLOADING
    document_type: DocumentType = DocumentType.GENERAL
    pages: int = 0
    chunks: int = 0
    parent_chunks: int = 0
    file_size_bytes: int = 0
    content_hash: str = ""
    ocr_used: bool = False
    ocr_confidence: Optional[float] = None
    language: str = "en"
    author: Optional[str] = None
    creation_date: Optional[str] = None
    title: Optional[str] = None
    error_message: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processing_completed_at: Optional[datetime] = None


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    document_type: DocumentType
    pages: int
    chunks: int
    ocr_used: bool
    uploaded_at: datetime
    processing_completed_at: Optional[datetime]
    error_message: Optional[str]


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class ChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int
    section: Optional[str]
    content: str
    chunk_type: str


# ── Query & Answer Schemas ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    document_ids: Optional[list[str]] = None  # None = all tenant docs
    conversation_id: Optional[str] = None
    enable_developer_mode: bool = False
    retrieval_config: Optional[RetrievalConfig] = None

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        return v.strip()


class RetrievalConfig(BaseModel):
    dense_top_k: int = Field(default=10, ge=1, le=50)
    bm25_top_k: int = Field(default=10, ge=1, le=50)
    rerank_top_k: int = Field(default=6, ge=1, le=20)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)


class Citation(BaseModel):
    citation_index: int
    document_id: str
    filename: str
    page: int
    chunk_id: str
    supporting_text: str
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ClaimVerification(BaseModel):
    claim: str
    supported: bool
    supporting_chunk_ids: list[str] = []
    evidence_text: Optional[str] = None
    failure_reason: Optional[str] = None


class AuditResult(BaseModel):
    status: AuditStatus
    claims_total: int
    claims_supported: int
    claims_unsupported: int
    citations_valid: int
    citations_invalid: int
    has_numerical_claims: bool
    numerical_claims_verified: bool
    has_contradictions: bool
    contradiction_details: Optional[str] = None
    confidence_score: float = 0.0
    grounded: bool = False
    failure_reasons: list[str] = []


class PipelineStage(BaseModel):
    """Developer mode pipeline stage data."""
    stage: str
    data: dict[str, Any]
    latency_ms: float = 0.0


class QueryAnalysis(BaseModel):
    intent: str
    complexity: str
    entities: list[str] = []
    keywords: list[str] = []
    document_constraints: list[str] = []
    requires_multi_document: bool = False
    requires_decomposition: bool = False
    requires_numerical_reasoning: bool = False
    route: QueryRoute = QueryRoute.SIMPLE


class QueryAnswer(BaseModel):
    answer_id: str
    question: str
    answer: str
    citations: list[Citation] = []
    audit: AuditResult
    query_analysis: Optional[QueryAnalysis] = None
    rewritten_queries: list[str] = []
    subqueries: list[str] = []
    pipeline_stages: list[PipelineStage] = []  # developer mode
    retry_count: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    conversation_id: Optional[str] = None
    tenant_id: str = ""


# ── Evidence / Dev Mode ───────────────────────────────────────────────────────

class EvidenceRequest(BaseModel):
    query_id: str


class EvidenceResponse(BaseModel):
    query_id: str
    retrieved_chunks: list[ChunkResponse]
    pipeline_stages: list[PipelineStage]


# ── Stats ─────────────────────────────────────────────────────────────────────

class SystemStats(BaseModel):
    total_documents: int
    total_pages: int
    total_chunks: int
    total_queries: int
    avg_latency_ms: float
    cache_hit_rate: float
    tenant_id: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
