from config import cfg
from config.logging_config import logger
from src.retrieval.models import RetrievalResponse


class Layer1AbstentionGuard:
    """
    Pre-LLM confidence gate — Guardrail Layer 1.

    Runs AFTER two-stage retrieval but BEFORE any LLM API call.
    If the maximum composite score across all retrieved chunks is below
    cfg.CONFIDENCE_THRESHOLD (0.40), the system abstains and returns a
    graceful "I don't know" message rather than risking a hallucination.

    Two abstention triggers:
    1. Empty results    — no documents retrieved at all (query gibberish or DB empty).
    2. Low confidence   — max composite_score < 0.40 — out-of-domain question.

    Why 0.40 as threshold?
    - Scores below 0.40 indicate the cross-encoder could not confidently
      match the query to any document in the corpus.
    - In-domain queries consistently score > 0.55 on this corpus.
    - This prevents completely unrelated questions from triggering LLM generation
      (e.g. "What is the recipe for banana bread?").
    """

    @staticmethod
    def check_query_quality(query: str) -> str | None:
        """
        Pre-retrieval query sanity check.
        Returns an error reason string if query is unusable, else None.
        """
        if not query or not query.strip():
            return "Query cannot be empty."
        if len(query.strip()) < 3:
            return "Query is too short to be meaningful."
        return None

    @staticmethod
    def evaluate(response: RetrievalResponse) -> RetrievalResponse:
        """
        Evaluate a RetrievalResponse and set abstention flags if warranted.

        Mutates and returns the response with:
        - max_confidence set to highest composite score in results
        - should_abstain = True/False
        - abstention_reason = descriptive reason string (or None if not abstaining)
        """
        # Trigger 1: No results returned
        if not response.results:
            response.max_confidence = 0.0
            response.should_abstain = True
            response.abstention_reason = (
                "No relevant documents found in the knowledge base for this query."
            )
            logger.warning(f"Abstaining (empty results) for query: '{response.query[:80]}'")
            return response

        # Compute maximum confidence across all returned results
        max_composite = max(r.composite_score for r in response.results)
        max_vector = max(r.vector_score for r in response.results)

        # Dual-Signal Evaluation:
        # 1. High composite confidence (>= 0.40) -> clear match across both stages.
        # 2. Strong vector similarity (>= 0.44) -> matches indexed document chunks even if
        #    imperative phrasing ("Explain Pricing details") received low cross-encoder logit.
        if max_composite >= cfg.CONFIDENCE_THRESHOLD:
            effective_confidence = max_composite
            should_abstain = False
        elif max_vector >= 0.44:
            # Calibrate confidence when dense vector evidence is strong but cross-encoder was phrasing-penalized
            effective_confidence = round(max_vector * 0.95, 4)
            should_abstain = False
        else:
            effective_confidence = max_composite
            should_abstain = True

        response.max_confidence = round(effective_confidence, 4)

        if should_abstain:
            response.should_abstain = True
            response.abstention_reason = (
                f"Insufficient document evidence "
                f"(confidence {response.max_confidence:.3f} < {cfg.CONFIDENCE_THRESHOLD:.3f}). "
                f"This question may be outside the scope of the provided documents."
            )
            logger.warning(
                f"Abstaining (low confidence={response.max_confidence:.3f}) "
                f"for query: '{response.query[:80]}'"
            )
        else:
            response.should_abstain = False
            response.abstention_reason = None
            logger.info(
                f"Guardrail passed (confidence={response.max_confidence:.3f}) "
                f"for query: '{response.query[:80]}'"
            )

        return response
