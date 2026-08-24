from typing import List, Optional
from pydantic import BaseModel, Field

from src.generation.models import RAGResponse


class EvaluationCase(BaseModel):
    """
    A single evaluation query test case with gold standard expectations.
    """
    id: str = Field(description="Unique query identifier, e.g. q-01")
    query: str = Field(description="The user question to ask the RAG pipeline")
    category: str = Field(
        description="Challenge category: direct_factual, table_reasoning, cross_doc, out_of_domain"
    )
    expected_docs: List[str] = Field(
        default_factory=list,
        description="List of document names expected to appear in retrieval/citations, e.g. ['Pricing_and_SLA']",
    )
    expected_facts: List[str] = Field(
        default_factory=list,
        description="Key terms, numbers, or phrases expected in the factual answer",
    )
    should_abstain: bool = Field(
        default=False,
        description="True if the query is out-of-domain and the system MUST abstain",
    )
    description: str = Field(
        default="",
        description="Brief explanation of what this test case verifies",
    )


class EvaluationResult(BaseModel):
    """
    Evaluation outcome for a single query test case.
    """
    case: EvaluationCase
    rag_response: RAGResponse
    retrieval_recall_pass: bool = Field(
        description="True if expected document was found in retrieved candidates"
    )
    citation_precision_pass: bool = Field(
        description="True if citations reference the correct document"
    )
    abstention_correct: bool = Field(
        description="True if system abstained when it should have, or answered when in-domain"
    )
    fact_match_score: float = Field(
        default=1.0,
        description="Fraction of expected fact keywords found in the generated answer (0.0 to 1.0)",
    )
    latency_ms: float = Field(description="End-to-end query latency in milliseconds")


class EvaluationSummary(BaseModel):
    """
    Aggregated benchmark metrics across all test cases.
    """
    total_queries: int
    in_domain_count: int
    out_of_domain_count: int
    retrieval_recall_rate: float
    citation_precision_rate: float
    grounded_fact_match_rate: float
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    results: List[EvaluationResult]
