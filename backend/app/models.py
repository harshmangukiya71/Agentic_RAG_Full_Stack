"""
models.py - shared Pydantic schemas used across the API.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single text chunk with full provenance metadata."""

    document: str
    page: int
    chunk_index: int
    text: str
    token_count: int
    section_title: Optional[str] = None
    entities: list[str] = Field(default_factory=list)
    kg_relations: list[dict[str, Any]] = Field(default_factory=list)
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    extraction_method: str = "native"


class RetrievedChunk(BaseModel):
    """A chunk augmented with a relevance score after retrieval/re-ranking."""

    document: str
    page: int
    chunk_index: int
    chunk: str
    relevance_score: float
    retrieval_source: str = "hybrid"
    entity_matches: list[str] = Field(default_factory=list)


class OCRBlock(BaseModel):
    text: str
    bbox: list[float] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class OCRPage(BaseModel):
    document_id: str
    page_number: int
    raw_text: str
    ocr_confidence: float = Field(1.0, ge=0.0, le=1.0)
    blocks: list[OCRBlock] = Field(default_factory=list)
    extraction_method: str = "native"


class EntityMention(BaseModel):
    entity_id: str
    text: str
    label: str
    normalized: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    document: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class GraphEntity(BaseModel):
    entity_id: str
    label: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=10)
    session_id: Optional[str] = Field(None, max_length=128)


class SourceReference(BaseModel):
    document: str
    page: int
    chunk: str
    chunk_index: Optional[int] = None
    entities: list[str] = Field(default_factory=list)


class QueryClassification(BaseModel):
    """Output of the query planning agent."""
    query_type: str = "LOOKUP"
    query_scope: str = "LOCAL"
    retrieval_strategy: str = "hybrid"
    top_k: int = 5
    reasoning_hints: list[str] = Field(default_factory=list)


class ReasoningOutput(BaseModel):
    """Structured evidence produced by the reasoning agent."""
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    calculations: list[dict[str, Any]] = Field(default_factory=list)
    rankings: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    evidence_sufficient: bool = True
    reasoning_steps: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    missing_relationships: list[str] = Field(default_factory=list)
    recommended_action: Optional[str] = None


class QueryResponse(BaseModel):
    status: str = "success"
    answer: str
    sources: list[SourceReference]
    confidence: float = Field(..., ge=0.0, le=1.0)
    query_classification: Optional[QueryClassification] = None
    original_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    cache_hit: bool = False
    cache_miss: bool = False
    latency_metrics: dict[str, float] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document: str
    pages_processed: int
    chunks_created: int
    summary: Optional[str] = None
    extraction_method: str = "native"
    average_ocr_confidence: Optional[float] = None
    entities_extracted: int = 0
    graph_relationships_created: int = 0
    status: str = "success"


class DocumentInfo(BaseModel):
    document: str
    total_chunks: int
    pages: list[int]
    summary: Optional[str] = None
    average_ocr_confidence: Optional[float] = None
    entities: list[str] = Field(default_factory=list)


class GraphNeighborsResponse(BaseModel):
    entity: GraphEntity | None = None
    neighbors: list[GraphEntity] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)


class EvalPair(BaseModel):
    question: str
    expected_document: str
    expected_page: int
    answer_hint: str = ""


class EvalResult(BaseModel):
    question: str
    expected_document: str
    expected_page: int
    retrieved_top5: list[dict]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    rank: int
    reciprocal_rank: float
    original_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0
    total_response_time_ms: float = 0.0


class EvalReport(BaseModel):
    total_questions: int
    hits_at_1: int
    hits_at_3: int
    hits_at_5: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    mrr: float
    precision_at_1: float = 0.0
    precision_at_3: float
    precision_at_5: float = 0.0
    hits: int
    retrieval_metrics: dict[str, Any] = Field(default_factory=dict)
    generation_metrics: dict[str, float] = Field(default_factory=dict)
    latency_metrics: dict[str, float] = Field(default_factory=dict)
    results: list[EvalResult]


class CacheStatus(BaseModel):
    """Cache loading progress after server restart."""
    cache_loaded_percent: int = 0
    is_ready: bool = False
    redis_available: bool = False
    total_entries: int = 0
    loaded_entries: int = 0


class CacheEntry(BaseModel):
    """A single cache entry for inspection."""
    question: str
    answer_preview: str
    ttl_seconds: int
    hits: int


class CacheEntriesResponse(BaseModel):
    """Response containing multiple cache entries."""
    entries: list[CacheEntry]


class ClearResponse(BaseModel):
    """Generic response for clear operations."""
    success: bool = True
    detail: str = ""
