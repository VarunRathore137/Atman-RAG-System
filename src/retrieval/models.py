from typing import List, Optional
from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """
    A single retrieved document chunk with full citation metadata and scores.

    Three scoring fields trace the chunk's journey:
    - vector_score:     1 - cosine_distance. From ChromaDB Stage 1 search.
    - rerank_score:     sigmoid(cross_encoder_logit). From Stage 2 reranking.
    - composite_score:  0.35 * vector_score + 0.65 * rerank_score.

    The composite_score is the definitive relevance ranking signal.
    """
    # Provenance metadata (mirrors EnrichedChunk fields for traceability)
    chunk_id: str
    doc_name: str
    doc_filename: str
    doc_code: str = "UNKNOWN"
    doc_version: str = "1.0"
    page_num: int
    section_heading: str = "General"
    chunk_type: str = "text"
    text: str
    has_table: bool = False
    has_code: bool = False

    # Scoring fields
    vector_score: float = 0.0       # [0, 1] — bi-encoder cosine similarity
    rerank_score: float = 0.0       # [0, 1] — cross-encoder sigmoid score
    composite_score: float = 0.0    # [0, 1] — weighted blend of above two
    rank: int = 1                   # 1-indexed position after reranking


class RetrievalResponse(BaseModel):
    """
    Complete response from the TwoStageRetriever pipeline.

    Carries both the ranked results and the Layer 1 guardrail verdict.
    The 'should_abstain' flag determines whether the LLM will be called
    (Phase 5) or the system returns a canned "I don't know" message.
    """
    query: str
    results: List[RetrievalResult] = []

    # Guardrail Layer 1 fields
    max_confidence: float = 0.0
    should_abstain: bool = False
    abstention_reason: Optional[str] = None

    # Performance tracking
    elapsed_ms: float = 0.0
