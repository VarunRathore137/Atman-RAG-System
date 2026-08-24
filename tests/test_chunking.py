import pytest

from config import cfg
from src.ingestion import PDFExtractor
from src.chunking import (
    SemanticChunker,
    EnrichedChunk,
    DOC_TYPE_MAP,
    QABoundarySplitter,
    TextSplitter,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def all_pages():
    """Extract all 7 PDFs once; reuse across tests in this module."""
    extractor = PDFExtractor(doc_dir=cfg.DOCS_DIR)
    return extractor.extract_all()


@pytest.fixture(scope="module")
def all_chunks(all_pages):
    """Chunk the full corpus once; reuse across tests."""
    chunker = SemanticChunker()
    return chunker.chunk_all(all_pages)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_chunk_count_range(all_chunks):
    """Corpus should produce a reasonable number of chunks (40–200 for 20 pages)."""
    count = len(all_chunks)
    print(f"\nTotal chunks: {count}")
    # Print breakdown for inspection
    by_type = {}
    for c in all_chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    print(f"By type: {by_type}")
    assert 40 <= count <= 200, f"Expected 40–200 chunks, got {count}"


def test_pricing_sla_tables_atomic(all_chunks):
    """
    Pricing_and_SLA must have at least 2 table chunks.
    Tables must be atomic (pipe chars present), contain all 3 tier names,
    and carry chunk_type='table'.
    """
    table_chunks = [
        c for c in all_chunks
        if c.doc_name == "Pricing_and_SLA" and c.has_table
    ]
    assert len(table_chunks) >= 2, f"Expected >=2 table chunks, got {len(table_chunks)}"

    combined = "\n".join(c.text for c in table_chunks)
    for tier in ("Free", "Standard", "Enterprise", "99.5%"):
        assert tier in combined, f"'{tier}' not found in Pricing_and_SLA table chunks"

    for tc in table_chunks:
        assert tc.chunk_type == "table"
        assert tc.has_table is True
        assert "|" in tc.text, "Markdown pipe character missing from table chunk"


def test_faq_support_qa_pair_chunks(all_chunks):
    """
    FAQ_Support pages must yield chunks with chunk_type='qa_pair'.
    Each chunk must belong to the 'qa_pairs' doc_type category.
    """
    faq_chunks = [c for c in all_chunks if c.doc_name == "FAQ_Support"]
    assert len(faq_chunks) >= 3, f"Expected >=3 FAQ chunks, got {len(faq_chunks)}"

    for c in faq_chunks:
        assert c.chunk_type == "qa_pair", (
            f"FAQ chunk has chunk_type='{c.chunk_type}', expected 'qa_pair'"
        )
        assert c.doc_type == "qa_pairs"


def test_chunk_metadata_schema(all_chunks):
    """
    Every chunk must satisfy the full metadata contract:
    - chunk_id non-empty
    - doc_filename ends with .pdf
    - char_count matches actual text length
    - token_count_est > 0
    - doc_type is a known value
    - to_chroma_metadata() returns only scalar values
    """
    known_doc_types = set(DOC_TYPE_MAP.values())

    for chunk in all_chunks:
        assert chunk.chunk_id, f"Empty chunk_id on chunk {chunk}"
        assert chunk.doc_name, "Empty doc_name"
        assert chunk.doc_filename.endswith(".pdf"), f"Bad filename: {chunk.doc_filename}"
        assert chunk.doc_code, "Empty doc_code"
        assert chunk.doc_version, "Empty doc_version"
        assert chunk.page_num >= 1, f"Invalid page_num: {chunk.page_num}"
        assert chunk.char_count == len(chunk.text), (
            f"char_count mismatch: stored={chunk.char_count} actual={len(chunk.text)}"
        )
        assert chunk.token_count_est > 0, "token_count_est must be > 0"
        assert chunk.doc_type in known_doc_types, (
            f"Unknown doc_type '{chunk.doc_type}' on {chunk.doc_name}"
        )

        # ChromaDB metadata must be flat scalars only
        meta = chunk.to_chroma_metadata()
        assert isinstance(meta, dict)
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Non-scalar metadata['{k}'] = {type(v).__name__} in chunk {chunk.chunk_id}"
            )


def test_qa_boundary_splitter_direct():
    """
    Unit test for QABoundarySplitter — verifies Q: boundary detection
    produces exactly 2 chunks from a 2-question FAQ sample.
    """
    sample = (
        "Q: How do I reset my password?\n"
        "Go to Settings and click Reset Password. An email will be sent.\n\n"
        "Q: Where is my data stored?\n"
        "Data is stored in AES-256 encrypted buckets in us-east-1."
    )
    chunks = QABoundarySplitter.split(sample)
    assert len(chunks) == 2, f"Expected 2 Q&A chunks, got {len(chunks)}: {chunks}"
    assert "How do I reset my password?" in chunks[0]
    assert "Where is my data stored?" in chunks[1]
