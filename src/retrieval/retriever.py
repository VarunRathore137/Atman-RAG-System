import time
from typing import Any, Dict, List, Optional

from config import cfg
from config.logging_config import logger
from src.vector_store.embedder import EmbeddingEngine
from src.vector_store.chroma_store import ChromaVectorStore
from src.retrieval.models import RetrievalResult, RetrievalResponse
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.guardrails import Layer1AbstentionGuard


class TwoStageRetriever:
    """
    Orchestrates the full two-stage retrieval pipeline:

    Stage 1 — Dense Vector Search (ChromaDB HNSW):
        Embeds the query via bi-encoder, retrieves top-K candidates by
        cosine similarity. Fast recall (~5ms), lower precision.
        vector_score = 1 - cosine_distance

    Stage 2 — Cross-Encoder Reranking:
        Scores every (query, candidate_text) pair with full bidirectional
        attention. Reorders candidates by precision. Slow but highly accurate.
        rerank_score  = sigmoid(cross_encoder_logit)
        composite_score = 0.35 * vector_score + 0.65 * rerank_score

    Layer 1 Guardrail:
        Checks max composite_score against CONFIDENCE_THRESHOLD (0.40).
        Sets should_abstain=True for out-of-domain queries before LLM call.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingEngine] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.embedder = embedder or EmbeddingEngine()
        self.vector_store = vector_store or ChromaVectorStore()
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        doc_filter: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        Execute the two-stage retrieval pipeline for a user query.

        Args:
            query:      User question string.
            top_k:      Stage 1 candidate pool size (default: cfg.RETRIEVAL_K = 20).
            top_n:      Stage 2 reranked results to return (default: cfg.RERANK_TOP_N = 5).
            where:      Raw ChromaDB metadata filter dict.
            doc_filter: Convenience shortcut — if set, adds where={"doc_name": doc_filter}.

        Returns:
            RetrievalResponse with ranked results and guardrail verdict.
        """
        start = time.perf_counter()
        k = top_k or cfg.RETRIEVAL_K
        n = top_n or cfg.RERANK_TOP_N

        # Pre-retrieval query quality check
        query_error = Layer1AbstentionGuard.check_query_quality(query)
        if query_error:
            elapsed = (time.perf_counter() - start) * 1000
            return RetrievalResponse(
                query=query,
                results=[],
                should_abstain=True,
                abstention_reason=query_error,
                elapsed_ms=round(elapsed, 2),
            )

        # Build metadata filter
        effective_where = where or {}
        if doc_filter:
            effective_where = {"doc_name": doc_filter}

        # ── Stage 1: Dense Vector Search ──────────────────────────────────
        logger.debug(f"Stage 1: querying ChromaDB top-{k} for '{query[:60]}'")
        query_embedding = self.embedder.embed_query(query)
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=min(k, self.vector_store.count()),
            where=effective_where if effective_where else None,
        )

        # Parse ChromaDB results into RetrievalResult objects
        candidates: List[RetrievalResult] = []
        if raw_results and raw_results.get("ids") and raw_results["ids"][0]:
            ids = raw_results["ids"][0]
            docs = raw_results["documents"][0]
            metas = raw_results["metadatas"][0]
            dists = raw_results["distances"][0]

            for chunk_id, text, meta, distance in zip(ids, docs, metas, dists):
                vector_score = max(0.0, 1.0 - float(distance))
                candidates.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        doc_name=meta.get("doc_name", ""),
                        doc_filename=meta.get("doc_filename", ""),
                        doc_code=meta.get("doc_code", "UNKNOWN"),
                        doc_version=meta.get("doc_version", "1.0"),
                        page_num=int(meta.get("page_num", 1)),
                        section_heading=meta.get("section_heading", "General"),
                        chunk_type=meta.get("chunk_type", "text"),
                        text=text,
                        has_table=bool(meta.get("has_table", False)),
                        has_code=bool(meta.get("has_code", False)),
                        vector_score=round(vector_score, 6),
                    )
                )

        # ── Stage 2: Cross-Encoder Reranking ──────────────────────────────
        if candidates:
            logger.debug(f"Stage 2: reranking {len(candidates)} candidates -> top-{n}")
            candidates = self.reranker.rerank(query, candidates, top_n=n)

        # ── Layer 1 Guardrail ──────────────────────────────────────────────
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response = RetrievalResponse(
            query=query,
            results=candidates,
            elapsed_ms=elapsed_ms,
        )
        response = Layer1AbstentionGuard.evaluate(response)

        logger.info(
            f"Retrieval complete in {elapsed_ms}ms: "
            f"{len(candidates)} results, "
            f"confidence={response.max_confidence:.3f}, "
            f"abstain={response.should_abstain}"
        )
        return response
