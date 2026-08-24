from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from src.retrieval.models import RetrievalResponse


class Citation(BaseModel):
    """
    Structured document citation with full provenance tracking.
    """
    doc_name: str
    doc_filename: str
    doc_code: str = "UNKNOWN"
    doc_version: str = "1.0"
    page_num: int
    section_heading: str = "General"
    chunk_type: str = "text"
    citation_string: str = ""  # e.g., "[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]"


class LLMResponse(BaseModel):
    """
    Standardized response from any LLM provider adapter (Groq, Ollama, Mock).
    """
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_ms: float = 0.0


class RAGResponse(BaseModel):
    """
    Final end-to-end response delivered to the user / UI.
    """
    query: str
    answer: str
    retrieval_response: Optional[RetrievalResponse] = None
    citations: List[Citation] = []
    confidence_score: float = 0.0
    confidence_badge: str = "UNKNOWN"  # "HIGH", "MEDIUM", "LOW", "ABSTAINED"
    is_abstained: bool = False
    abstention_reason: Optional[str] = None
    provider: str = "unknown"
    model_name: str = "unknown"
    total_latency_ms: float = 0.0
    validation_warnings: List[str] = []
