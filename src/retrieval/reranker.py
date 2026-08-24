import math
from typing import List, Optional

from sentence_transformers.cross_encoder import CrossEncoder

from config import cfg
from config.logging_config import logger
from src.retrieval.models import RetrievalResult


class CrossEncoderReranker:
    """
    Stage 2 of the two-stage retrieval pipeline.

    Uses a cross-encoder (BERT-based) to score each (query, chunk) pair with
    full bidirectional attention. This captures token-to-token interactions
    missed by the bi-encoder (Stage 1) and dramatically improves precision.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Trained on MS MARCO passage ranking benchmark
      - Outputs raw logits (unbounded real values)
      - Converted to [0, 1] via sigmoid: sigmoid(x) = 1 / (1 + e^(-x))

    Composite scoring formula:
      composite = VECTOR_WEIGHT * vector_score + RERANK_WEIGHT * rerank_score
                = 0.35 * (1 - cosine_distance) + 0.65 * sigmoid(logit)

    Why 65% weight on cross-encoder vs 35% on bi-encoder?
    Cross-encoders are empirically more accurate but slower (O(n) inference
    per candidate). The blend provides bias correction from both signals.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or cfg.RERANKER_MODEL
        self.device = device or "cpu"
        logger.info(f"Loading cross-encoder '{self.model_name}'...")
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Cross-encoder loaded.")

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid for unbounded logit -> [0, 1] score."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            return ex / (1.0 + ex)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank candidates using full cross-attention scoring.

        Args:
            query:      The user's question string.
            candidates: List of RetrievalResult from Stage 1 dense search.
            top_n:      How many to return after reranking (default: cfg.RERANK_TOP_N).

        Returns:
            Candidates sorted descending by composite_score, sliced to top_n.
        """
        if not candidates:
            return []

        n = top_n or cfg.RERANK_TOP_N

        # Build query-document pairs for batch cross-encoder inference
        pairs = [(query, c.text) for c in candidates]

        # Predict raw logits for all pairs in one batch
        logits = self._model.predict(pairs)

        # Score and update each candidate
        for candidate, logit in zip(candidates, logits):
            rerank_score = self._sigmoid(float(logit))
            composite = (
                cfg.VECTOR_WEIGHT * candidate.vector_score
                + cfg.RERANK_WEIGHT * rerank_score
            )
            candidate.rerank_score = round(rerank_score, 6)
            candidate.composite_score = round(composite, 6)

        # Sort descending by composite score
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        # Assign rank and return top-n
        for i, c in enumerate(candidates):
            c.rank = i + 1

        return candidates[:n]
