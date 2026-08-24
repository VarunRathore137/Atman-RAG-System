import argparse
import time
from typing import Any, Dict, Optional

from config import cfg
from config.logging_config import logger
from src.ingestion import PDFExtractor
from src.chunking import SemanticChunker
from src.vector_store.embedder import EmbeddingEngine
from src.vector_store.chroma_store import ChromaVectorStore


class CorpusIndexer:
    """
    Orchestrates the full PDF-to-ChromaDB indexing pipeline:
        1. Extract: PDFExtractor  -> List[PageDocument]     (Phase 1)
        2. Chunk:   SemanticChunker -> List[EnrichedChunk]  (Phase 2)
        3. Embed:   EmbeddingEngine -> List[List[float]]    (Phase 3)
        4. Upsert:  ChromaVectorStore -> persistent index   (Phase 3)

    Idempotency:
        - Without --force: if vectors already exist, skip re-indexing.
          Startup cost for Streamlit becomes ~0ms after first run.
        - With --force: reset collection, re-embed from scratch.
    """

    def __init__(
        self,
        extractor: Optional[PDFExtractor] = None,
        chunker: Optional[SemanticChunker] = None,
        embedder: Optional[EmbeddingEngine] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ):
        self.extractor = extractor or PDFExtractor(doc_dir=cfg.DOCS_DIR)
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or EmbeddingEngine()
        self.vector_store = vector_store or ChromaVectorStore()

    def index_corpus(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Run the full ingestion-to-index pipeline.

        Args:
            force_reindex: If True, wipe the collection and rebuild from scratch.
                           If False and vectors exist, return existing state instantly.

        Returns:
            Metrics dict with doc_count, page_count, chunk_count, vector_count, elapsed_sec.
        """
        start = time.time()

        # ── Idempotency check ──────────────────────────────────────────────
        existing = self.vector_store.count()
        if existing > 0 and not force_reindex:
            elapsed = time.time() - start
            logger.info(
                f"Collection already has {existing} vectors. "
                f"Skipping re-index (use --force to rebuild)."
            )
            return {
                "doc_count": 7,
                "chunk_count": existing,
                "vector_count": existing,
                "elapsed_sec": round(elapsed, 3),
                "skipped": True,
            }

        # ── Force reset if requested ───────────────────────────────────────
        if force_reindex and existing > 0:
            logger.info("Force reindex: resetting collection...")
            self.vector_store.reset_collection()

        # ── Step 1: PDF Extraction ─────────────────────────────────────────
        logger.info("Step 1/4: Extracting PDFs...")
        pages = self.extractor.extract_all()
        doc_count = len({p.doc_name for p in pages})
        page_count = len(pages)

        # ── Step 2: Semantic Chunking ──────────────────────────────────────
        logger.info("Step 2/4: Chunking pages...")
        chunks = self.chunker.chunk_all(pages)
        chunk_count = len(chunks)

        # ── Step 3: Dense Embedding ────────────────────────────────────────
        logger.info(f"Step 3/4: Embedding {chunk_count} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # ── Step 4: Upsert to ChromaDB ─────────────────────────────────────
        logger.info("Step 4/4: Upserting vectors into ChromaDB...")
        vector_count = self.vector_store.upsert_chunks(chunks, embeddings)

        elapsed = time.time() - start
        metrics = {
            "doc_count": doc_count,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "vector_count": vector_count,
            "elapsed_sec": round(elapsed, 2),
            "skipped": False,
        }
        logger.info(f"Indexing complete in {elapsed:.2f}s: {metrics}")
        return metrics


# ── CLI Entrypoint ─────────────────────────────────────────────────────────────

def _print_summary(metrics: Dict[str, Any]) -> None:
    if metrics.get("skipped"):
        print("\n[CorpusIndexer] Already indexed — skipped re-indexing.")
        print(f"  Vectors in collection: {metrics['vector_count']}")
        print("  Run with --force to rebuild from scratch.")
    else:
        print("\n[CorpusIndexer] Indexing complete!")
        print(f"  Documents   : {metrics['doc_count']}")
        print(f"  Pages       : {metrics.get('page_count', '?')}")
        print(f"  Chunks      : {metrics['chunk_count']}")
        print(f"  Vectors     : {metrics['vector_count']}")
        print(f"  Elapsed     : {metrics['elapsed_sec']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index the Atman Cloud document corpus into ChromaDB."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if vectors already exist.",
    )
    args = parser.parse_args()

    indexer = CorpusIndexer()
    metrics = indexer.index_corpus(force_reindex=args.force)
    _print_summary(metrics)
