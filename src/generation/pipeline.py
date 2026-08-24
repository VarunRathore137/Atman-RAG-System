import time
from typing import Optional

from config.logging_config import logger
from src.retrieval.retriever import TwoStageRetriever
from src.generation.models import RAGResponse
from src.generation.prompt_builder import PromptBuilder
from src.generation.validator import ResponseValidator
from src.generation.llm_client import BaseLLMClient, LLMClientFactory


class RAGPipeline:
    """
    End-to-End Enterprise RAG Pipeline Orchestrator.

    Integrates:
    - Stage 1 Dense Vector Search (ChromaDB)
    - Stage 2 Cross-Encoder Precision Reranking (ms-marco-MiniLM)
    - Guardrail Layer 1: Pre-LLM Confidence Abstention Gate (< 0.40)
    - Grounded Prompt Construction with Strict Excerpt Provenance
    - Multi-Provider LLM Generation (Groq / Ollama / Mock)
    - Guardrail Layer 2: Post-Generation URL Sanitizer & Citation Validator
    """

    def __init__(
        self,
        retriever: Optional[TwoStageRetriever] = None,
        llm_client: Optional[BaseLLMClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[ResponseValidator] = None,
    ):
        self.retriever = retriever or TwoStageRetriever()
        self.llm_client = llm_client or LLMClientFactory.get_client()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.validator = validator or ResponseValidator()

    @staticmethod
    def _compute_badge(confidence_score: float, is_abstained: bool) -> str:
        """Map numeric confidence score to UI badge category."""
        if is_abstained or confidence_score < 0.40:
            return "ABSTAINED"
        if confidence_score >= 0.70:
            return "HIGH"
        if confidence_score >= 0.50:
            return "MEDIUM"
        return "LOW"

    def query(
        self,
        query_str: str,
        doc_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute the complete RAG query pipeline.

        Args:
            query_str:   User question.
            doc_filter:  Optional document name filter (e.g. 'FAQ_Support').
            top_k:       Candidate pool size for Stage 1 (default: 20).
            top_n:       Reranked results count for Stage 2 (default: 5).
            temperature: LLM generation temperature.
            max_tokens:  LLM generation max token limit.

        Returns:
            Structured RAGResponse with grounded answer, citations, and confidence badge.
        """
        start_time = time.perf_counter()
        logger.info(f"RAGPipeline: Processing query '{query_str[:80]}'")

        # ── Step 1: Two-Stage Retrieval & Guardrail Layer 1 ────────────────
        retrieval_response = self.retriever.retrieve(
            query=query_str,
            top_k=top_k,
            top_n=top_n,
            doc_filter=doc_filter,
        )

        confidence_score = retrieval_response.max_confidence

        # ── Step 2: Check Pre-LLM Abstention Gate ─────────────────────────
        if retrieval_response.should_abstain:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            abstention_msg = (
                retrieval_response.abstention_reason
                or "Based on the provided Atman Cloud documentation, there is insufficient evidence to answer this question."
            )
            logger.info(
                f"RAGPipeline: Abstaining without LLM call (confidence={confidence_score:.3f})"
            )
            return RAGResponse(
                query=query_str,
                answer=abstention_msg,
                retrieval_response=retrieval_response,
                citations=[],
                confidence_score=confidence_score,
                confidence_badge="ABSTAINED",
                is_abstained=True,
                abstention_reason=retrieval_response.abstention_reason,
                provider=getattr(self.llm_client, "provider", "unknown"),
                model_name=getattr(self.llm_client, "model", "unknown"),
                total_latency_ms=round(elapsed_ms, 2),
                validation_warnings=[],
            )

        # ── Step 3: Construct Grounded Chat Messages ──────────────────────
        messages = self.prompt_builder.build_chat_messages(
            query=query_str,
            chunks=retrieval_response.results,
        )

        # ── Step 4: LLM Generation ────────────────────────────────────────
        llm_response = self.llm_client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # ── Step 5: Guardrail Layer 2 Validation & Citation Formatting ───
        cleaned_answer, citations, warnings = self.validator.validate(
            answer=llm_response.content,
            context_chunks=retrieval_response.results,
        )

        badge = self._compute_badge(confidence_score, is_abstained=False)
        total_latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"RAGPipeline: Query completed in {total_latency_ms:.2f}ms "
            f"[Badge: {badge} | Citations: {len(citations)} | Confidence: {confidence_score:.3f}]"
        )

        return RAGResponse(
            query=query_str,
            answer=cleaned_answer,
            retrieval_response=retrieval_response,
            citations=citations,
            confidence_score=confidence_score,
            confidence_badge=badge,
            is_abstained=False,
            abstention_reason=None,
            provider=llm_response.provider,
            model_name=llm_response.model,
            total_latency_ms=round(total_latency_ms, 2),
            validation_warnings=warnings,
        )
