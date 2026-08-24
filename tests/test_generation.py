"""
Phase 5 integration and end-to-end pipeline tests:
PromptBuilder, ResponseValidator, LLMClientFactory, and RAGPipeline.
"""

import pytest

from src.retrieval.models import RetrievalResult
from src.generation.models import Citation, RAGResponse
from src.generation.prompt_builder import PromptBuilder
from src.generation.validator import ResponseValidator
from src.generation.llm_client import MockLLMClient, LLMClientFactory
from src.generation.pipeline import RAGPipeline


# ── Shared Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_llm_client():
    """Deterministic local mock LLM client for automated testing."""
    return MockLLMClient()


@pytest.fixture(scope="module")
def pipeline(mock_llm_client):
    """Shared RAGPipeline instance with MockLLMClient."""
    return RAGPipeline(llm_client=mock_llm_client)


@pytest.fixture
def sample_retrieval_chunks():
    """Sample retrieval results matching real corpus schema."""
    c1 = RetrievalResult(
        chunk_id="Pricing_and_SLA__p002__t01",
        doc_name="Pricing_and_SLA",
        doc_filename="Pricing_and_SLA.pdf",
        doc_code="PRC-SLA-021",
        doc_version="3.2",
        page_num=2,
        section_heading="Pricing Tiers Table",
        chunk_type="table",
        text="| Tier | Price | Storage |\n| Free | $0 | 5 GB |\n| Standard | $12/user/mo | 500 GB |\n| Enterprise | Custom | Unlimited |",
        has_table=True,
        has_code=False,
        vector_score=0.75,
        rerank_score=0.92,
        composite_score=0.86,
        rank=1,
    )
    c2 = RetrievalResult(
        chunk_id="Pricing_and_SLA__p002__t02",
        doc_name="Pricing_and_SLA",
        doc_filename="Pricing_and_SLA.pdf",
        doc_code="PRC-SLA-021",
        doc_version="3.2",
        page_num=2,
        section_heading="SLA Guarantees",
        chunk_type="table",
        text="| Tier | Uptime |\n| Standard | 99.5% |\n| Enterprise | 99.95% |",
        has_table=True,
        has_code=False,
        vector_score=0.70,
        rerank_score=0.85,
        composite_score=0.7975,
        rank=2,
    )
    return [c1, c2]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_prompt_builder_grounded_format(sample_retrieval_chunks):
    """
    PromptBuilder must inject strict grounding rules and structured provenance headers.
    """
    messages = PromptBuilder.build_chat_messages(
        query="What is the pricing for the Standard tier?",
        chunks=sample_retrieval_chunks,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "STRICT GROUNDING" in messages[0]["content"]
    assert "CITATION FORMAT" in messages[0]["content"]

    user_msg = messages[1]["content"]
    assert "PRC-SLA-021" in user_msg
    assert "Pricing_and_SLA" in user_msg
    assert "Page: 2" in user_msg
    assert "USER QUESTION: What is the pricing for the Standard tier?" in user_msg


def test_response_validator_scrubs_hallucinated_urls(sample_retrieval_chunks):
    """
    ResponseValidator must scrub URLs that are not present in retrieved context.
    """
    raw_answer = (
        "The standard tier costs $12. For more info visit https://malicious-fake-site.com/docs "
        "and read our guide."
    )
    cleaned, citations, warnings = ResponseValidator.validate(
        answer=raw_answer,
        context_chunks=sample_retrieval_chunks,
    )

    assert "https://malicious-fake-site.com/docs" not in cleaned
    assert "[URL not in source documentation]" in cleaned
    assert len(warnings) == 1
    assert "Scrubbed hallucinated URL" in warnings[0]


def test_response_validator_citation_deduplication(sample_retrieval_chunks):
    """
    Two chunks from the same document and page must produce a single deduplicated Citation.
    """
    citations = ResponseValidator.build_citations(sample_retrieval_chunks)

    assert len(citations) == 1, (
        f"Expected 1 deduplicated citation for (Pricing_and_SLA, Page 2), got {len(citations)}"
    )
    c = citations[0]
    assert c.doc_name == "Pricing_and_SLA"
    assert c.doc_code == "PRC-SLA-021"
    assert c.doc_version == "3.2"
    assert c.page_num == 2
    assert c.citation_string == "[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]"


def test_rag_pipeline_in_domain_query(pipeline):
    """
    End-to-end query for subscription pricing tiers:
    Must return a grounded answer, citations list, confidence score >= 0.50, and HIGH or MEDIUM badge.
    """
    resp = pipeline.query("What are the subscription pricing tiers and monthly cost?")

    assert not resp.is_abstained
    assert resp.confidence_score >= 0.50
    assert resp.confidence_badge in ["HIGH", "MEDIUM"]
    assert len(resp.citations) > 0
    assert any(c.doc_name == "Pricing_and_SLA" for c in resp.citations)
    assert len(resp.answer) > 20
    assert resp.total_latency_ms > 0

    print(f"\nIn-Domain Query Test: Badge={resp.confidence_badge}, Score={resp.confidence_score:.4f}")
    print(f"Citations: {[c.citation_string for c in resp.citations]}")
    print(f"Answer: {resp.answer}")


def test_rag_pipeline_out_of_domain_abstention(pipeline):
    """
    End-to-end query for out-of-domain topic:
    Must trigger pre-LLM Layer 1 guardrail abstention (is_abstained=True, badge='ABSTAINED', 0 citations).
    """
    resp = pipeline.query(
        "What is the authentic recipe for Italian chocolate tiramisu cake?"
    )

    assert resp.is_abstained
    assert resp.confidence_badge == "ABSTAINED"
    assert resp.confidence_score < 0.40
    assert len(resp.citations) == 0
    assert resp.abstention_reason is not None
    assert "Insufficient document evidence" in resp.answer or "not available" in resp.answer

    print(f"\nOut-of-Domain Abstention Test: Badge={resp.confidence_badge}, Score={resp.confidence_score:.4f}")
    print(f"Abstention Message: {resp.answer}")


def test_rag_pipeline_doc_filter(pipeline):
    """
    Query with doc_filter='FAQ_Support':
    Must constrain retrieval and citations exclusively to FAQ_Support document.
    """
    resp = pipeline.query(
        "How do I reset my account password?",
        doc_filter="FAQ_Support",
    )

    assert not resp.is_abstained
    assert len(resp.citations) > 0
    for citation in resp.citations:
        assert citation.doc_name == "FAQ_Support", (
            f"Doc filter violated: got citation {citation.doc_name}"
        )

    print(f"\nDoc Filter Test: All citations from FAQ_Support: {[c.citation_string for c in resp.citations]}")
