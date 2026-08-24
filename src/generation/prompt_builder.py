from typing import Dict, List

from src.retrieval.models import RetrievalResult


class PromptBuilder:
    """
    Constructs grounded chat messages for LLM generation.

    Enforces strict hallucination prevention:
    1. The model is constrained to answer solely using the provided excerpts.
    2. The model is instructed to cite sources using:
       [Document Name (DOC_CODE vX.X), Page N]
    3. If the context does not contain sufficient information, the model
       is instructed to acknowledge the limitation rather than guessing.
    """

    SYSTEM_PROMPT = """You are the official enterprise AI assistant for Atman Cloud documentation.
Your job is to answer user queries accurately, concisely, and factually based ONLY on the provided context excerpts below.

CRITICAL RULES:
1. STRICT GROUNDING: Answer using ONLY the facts directly mentioned in the context excerpts. Do NOT assume, extrapolate, or bring in outside knowledge.
2. ABSENCE OF EVIDENCE: If the provided excerpts do not contain the answer, clearly state: "Based on the provided Atman Cloud documentation, this information is not available."
3. CITATION FORMAT: After every key statement or factual claim, cite the source using the exact format:
   [Document_Name (DOC_CODE vX.X), Page N]
   Example: The Enterprise plan includes 24/7 dedicated support [Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2].
4. TABULAR DATA: When answering questions regarding pricing, SLA percentages, or specs, extract exact numbers from the Markdown tables provided.
5. NO FABRICATED URLS: Never invent or assume URLs, links, or email addresses that do not explicitly appear in the excerpts.
"""

    @classmethod
    def format_context(cls, chunks: List[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a clean, structured context string with provenance headers.
        """
        if not chunks:
            return "No context excerpts available."

        formatted_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = (
                f"--- EXCERPT {i} ---\n"
                f"Document: {chunk.doc_name} ({chunk.doc_code} v{chunk.doc_version})\n"
                f"Page: {chunk.page_num} | Section: {chunk.section_heading} | Type: {chunk.chunk_type}\n"
            )
            content = f"{header}\n{chunk.text.strip()}\n"
            formatted_parts.append(content)

        return "\n".join(formatted_parts)

    @classmethod
    def build_chat_messages(
        cls, query: str, chunks: List[RetrievalResult]
    ) -> List[Dict[str, str]]:
        """
        Build standard OpenAI/Groq compatible chat messages.
        """
        context_str = cls.format_context(chunks)
        user_content = (
            f"DOCUMENT CONTEXT EXCERPTS:\n"
            f"========================\n"
            f"{context_str}\n"
            f"========================\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Please provide a clear, factual answer citing the relevant document excerpts."
        )

        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content},
        ]
