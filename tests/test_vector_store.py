"""
Phase 3 integration tests: EmbeddingEngine, ChromaVectorStore, CorpusIndexer.

These tests use the REAL embedded corpus (indexed once via module-scoped fixture)
to verify:
1. Embedding dimension correctness (384-dim)
2. ChromaDB upsert and count integrity
3. Semantic recall — pricing query retrieves Pricing_and_SLA table chunks
4. Metadata filtering — where clause constrains results correctly
5. Idempotency — second call without --force does not duplicate vectors
"""

import pytest

from src.vector_store.embedder import EmbeddingEngine
from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.indexer import CorpusIndexer


# ── Module-scope fixtures (index corpus ONCE, reuse across all 5 tests) ────────

@pytest.fixture(scope="module")
def embedder():
    """Shared EmbeddingEngine instance — model loaded once."""
    return EmbeddingEngine()


@pytest.fixture(scope="module")
def vector_store():
    """
    Dedicated test ChromaDB collection ('test_atman_docs') to avoid
    touching the production 'atman_docs' collection during testing.
    """
    return ChromaVectorStore(collection_name="test_atman_docs")


@pytest.fixture(scope="module")
def indexed_store(embedder, vector_store):
    """
    Index all 7 PDFs into the test collection (once per test session).
    Returns the metrics dict from CorpusIndexer.index_corpus(force_reindex=True).
    """
    indexer = CorpusIndexer(embedder=embedder, vector_store=vector_store)
    metrics = indexer.index_corpus(force_reindex=True)
    return metrics, vector_store, embedder


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_embedding_dimensions(embedder):
    """EmbeddingEngine must produce 384-dimensional vectors."""
    assert embedder.dimension == 384

    # Single query
    vec = embedder.embed_query("What is the pricing for the Standard plan?")
    assert len(vec) == 384
    assert isinstance(vec[0], float)

    # Batch of 3
    batch = embedder.embed_texts(["Hello", "World", "RAG pipeline"])
    assert len(batch) == 3
    assert all(len(v) == 384 for v in batch)


def test_chroma_store_upsert_and_count(indexed_store):
    """After indexing, vector count must match chunk count (~47)."""
    _, vs, _ = indexed_store
    count = vs.count()
    # We produced 47 chunks from 20 pages across 7 docs
    assert count >= 40, f"Expected >= 40 vectors, got {count}"
    assert count <= 80, f"Expected <= 80 vectors, got {count}"
    print(f"\nVector count in test collection: {count}")


def test_semantic_query_retrieval(indexed_store):
    """
    Querying 'pricing tiers monthly cost' must return Pricing_and_SLA table chunks
    in the top-5 results. This validates semantic recall on structured tabular data.
    """
    _, vs, emb = indexed_store
    query_vec = emb.embed_query("What are the subscription pricing tiers and monthly cost?")
    results = vs.query(query_vec, n_results=5)

    doc_names = [m["doc_name"] for m in results["metadatas"][0]]
    chunk_types = [m["chunk_type"] for m in results["metadatas"][0]]

    print(f"\nTop-5 results for pricing query: {list(zip(doc_names, chunk_types))}")

    assert "Pricing_and_SLA" in doc_names, (
        f"Expected Pricing_and_SLA in top-5 results, got: {doc_names}"
    )


def test_metadata_filtering(indexed_store):
    """
    Querying with where={'doc_name': 'FAQ_Support'} must only return FAQ chunks.
    Validates ChromaDB metadata pre-filtering works correctly.
    """
    _, vs, emb = indexed_store
    query_vec = emb.embed_query("reset password account login")
    results = vs.query(
        query_vec,
        n_results=5,
        where={"doc_name": "FAQ_Support"},
    )

    returned_docs = [m["doc_name"] for m in results["metadatas"][0]]
    print(f"\nFiltered results (FAQ_Support only): {returned_docs}")

    assert all(d == "FAQ_Support" for d in returned_docs), (
        f"Metadata filter failed: got docs {set(returned_docs)}"
    )


def test_indexer_idempotency(embedder, vector_store):
    """
    Running index_corpus() without force_reindex=True must NOT duplicate vectors.
    Count before == count after second call.
    """
    count_before = vector_store.count()
    assert count_before > 0, "Collection must be populated before testing idempotency"

    indexer = CorpusIndexer(embedder=embedder, vector_store=vector_store)
    metrics = indexer.index_corpus(force_reindex=False)

    count_after = vector_store.count()
    assert count_after == count_before, (
        f"Idempotency broken: count went from {count_before} to {count_after}"
    )
    assert metrics.get("skipped") is True, "Expected skipped=True for non-forced run"
    print(f"\nIdempotency confirmed: count stayed at {count_after}")
