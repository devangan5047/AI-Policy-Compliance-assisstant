from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PolicyMetadata(BaseModel):
    source: str = "uploaded-policy"
    title: str | None = None
    category: str = "internal"
    framework: str = "internal"
    clause_id: str | None = None
    page: int | None = None
    authority: int = Field(default=5, ge=1, le=10)
    effective_date: str | None = None


class PolicyChunk(BaseModel):
    id: str
    text: str
    metadata: PolicyMetadata
    score: float = 0


class QueryFilters(BaseModel):
    category: str | None = None
    framework: str | None = None
    source: str | None = None


class PolicyQueryRequest(BaseModel):
    question: str = Field(..., min_length=8, max_length=2000)
    employee_context: str | None = Field(default=None, max_length=2000)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    top_k: int = Field(default=6, ge=1, le=20)


class Citation(BaseModel):
    source: str
    title: str | None = None
    framework: str
    category: str
    clause_id: str | None = None
    page: int | None = None
    snippet: str
    score: float


class AgentTrace(BaseModel):
    agent: str
    summary: str
    confidence: float = Field(ge=0, le=1)


class ComplianceAnswer(BaseModel):
    answer: str
    compliance_status: Literal["compliant", "non_compliant", "needs_review", "insufficient_context"]
    compliance_score: int = Field(ge=0, le=100)
    risk_level: Literal["low", "medium", "high", "critical"]
    escalation_required: bool
    citations: list[Citation]
    conflicts: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    agent_trace: list[AgentTrace] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    retrieved_context_count: int = 0


class UploadResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    metadata: PolicyMetadata


class BatchUploadResponse(BaseModel):
    documents: list[UploadResponse]
    total_documents: int
    total_chunks_indexed: int


class AnalyticsEvent(BaseModel):
    question: str
    status: str
    risk_level: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    framework: str | None = None
    category: str | None = None


class BenchmarkResult(BaseModel):
    framework: str
    query_count: int
    average_latency_ms: float
    average_context_count: float
    notes: str | None = None
