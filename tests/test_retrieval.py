"""
Phase 4 integration tests: TwoStageRetriever, CrossEncoderReranker,
Layer1AbstentionGuard.

Uses the REAL indexed ChromaDB collection and real embedding models.
All fixtures are module-scoped so models load only once per test session.
"""

import pytest

from src.retrieval.models import RetrievalResult, RetrievalResponse
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.guardrails import Layer1AbstentionGuard
from src.retrieval.retriever import TwoStageRetriever


# ── Module-scope shared fixtures ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def retriever():
    """Shared TwoStageRetriever instance (loads both models once)."""
    return TwoStageRetriever()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_two_stage_retrieval_pricing(retriever):
    """
    End-to-end retrieval for a pricing question.
    Expected: Rank 1 result from Pricing_and_SLA with high confidence (>= 0.50).
    """
    resp = retriever.retrieve(
        "What are the subscription pricing tiers and monthly cost?"
    )

    assert len(resp.results) > 0, "Expected at least 1 result"
    assert resp.results[0].doc_name == "Pricing_and_SLA", (
        f"Expected Pricing_and_SLA at rank 1, got: {resp.results[0].doc_name}"
    )
    assert resp.max_confidence >= 0.50, (
        f"Expected confidence >= 0.50 for pricing query, got: {resp.max_confidence}"
    )
    assert not resp.should_abstain, "Should not abstain for an in-domain pricing query"

    print(f"\nPricing query - Top doc: {resp.results[0].doc_name}")
    print(f"  Composite score: {resp.results[0].composite_score:.4f}")
    print(f"  Vector score: {resp.results[0].vector_score:.4f}")
    print(f"  Rerank score: {resp.results[0].rerank_score:.4f}")
    print(f"  Elapsed: {resp.elapsed_ms}ms")


def test_stage2_reranker_reordering(retriever):
    """
    Cross-Encoder must meaningfully assign rerank_score to all results.
    All top-N results should have rerank_score > 0 (cross-encoder was active).
    """
    resp = retriever.retrieve("What is the SLA uptime guarantee for Enterprise plan?")

    assert len(resp.results) > 0, "Expected at least 1 result"
    for r in resp.results:
        assert r.rerank_score > 0.0, (
            f"Cross-encoder reranker was not applied: rerank_score=0 on {r.chunk_id}"
        )
        assert 0.0 <= r.composite_score <= 1.0, (
            f"Composite score out of [0,1] range: {r.composite_score}"
        )
        assert r.rank >= 1, f"Invalid rank: {r.rank}"

    # Verify results are sorted descending by composite score
    scores = [r.composite_score for r in resp.results]
    assert scores == sorted(scores, reverse=True), (
        f"Results are not sorted by composite score: {scores}"
    )


def test_composite_confidence_score_formula():
    """
    Unit test: composite_score = 0.35 * vector_score + 0.65 * rerank_score.
    Verifies the exact math formula within floating point tolerance.
    """
    from config import cfg

    rr = CrossEncoderReranker()
    c1 = RetrievalResult(
        chunk_id="t1", doc_name="d", doc_filename="f.pdf", page_num=1,
        text="The Standard plan costs $12 per user per month.",
        vector_score=0.80,
    )
    c2 = RetrievalResult(
        chunk_id="t2", doc_name="d", doc_filename="f.pdf", page_num=1,
        text="Password reset is available under account settings.",
        vector_score=0.60,
    )

    results = rr.rerank("What does the Standard plan cost per month?", [c1, c2])

    for r in results:
        expected = cfg.VECTOR_WEIGHT * r.vector_score + cfg.RERANK_WEIGHT * r.rerank_score
        assert abs(r.composite_score - expected) < 1e-4, (
            f"Composite formula mismatch: got {r.composite_score}, expected {expected:.6f}"
        )
        assert 0.0 <= r.rerank_score <= 1.0, f"rerank_score out of bounds: {r.rerank_score}"

    print(f"\nFormula verified: top={results[0].chunk_id} composite={results[0].composite_score:.4f}")


def test_layer1_abstention_in_vs_out_of_domain(retriever):
    """
    Layer 1 Guardrail behaviour:
    - In-domain query → should_abstain == False, max_confidence >= 0.40
    - Out-of-domain query → should_abstain == True, max_confidence < 0.40
    """
    # In-domain: documented in Pricing_and_SLA
    in_domain = retriever.retrieve("What is the Enterprise plan uptime guarantee?")
    print(f"\nIn-domain confidence: {in_domain.max_confidence:.4f}, abstain: {in_domain.should_abstain}")
    assert not in_domain.should_abstain, (
        f"Should NOT abstain for in-domain query. "
        f"confidence={in_domain.max_confidence:.4f}"
    )
    assert in_domain.max_confidence >= 0.40

    # Out-of-domain: nothing about cake recipes in the corpus
    out_of_domain = retriever.retrieve(
        "What is the traditional Italian recipe for tiramisu dessert cake?"
    )
    print(f"Out-of-domain confidence: {out_of_domain.max_confidence:.4f}, abstain: {out_of_domain.should_abstain}")
    assert out_of_domain.should_abstain, (
        f"Should abstain for out-of-domain query. "
        f"confidence={out_of_domain.max_confidence:.4f}"
    )
    assert out_of_domain.abstention_reason is not None


def test_metadata_filtered_retrieval(retriever):
    """
    Retrieval with doc_filter='FAQ_Support' must constrain all results to FAQ chunks.
    Validates metadata pre-filtering integration with the two-stage pipeline.
    """
    resp = retriever.retrieve(
        "How do I reset my account password?",
        doc_filter="FAQ_Support",
    )

    assert len(resp.results) > 0, "Expected results for FAQ_Support filter"
    doc_names = {r.doc_name for r in resp.results}
    assert doc_names == {"FAQ_Support"}, (
        f"Metadata filter failed: got docs {doc_names}"
    )
    print(f"\nDoc filter verified: all {len(resp.results)} results from FAQ_Support")
