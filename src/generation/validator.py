import re
from typing import List, Set, Tuple

from config.logging_config import logger
from src.generation.models import Citation
from src.retrieval.models import RetrievalResult


class ResponseValidator:
    """
    Guardrail Layer 2: Post-Generation Response Validator.

    Performs safety and integrity checks on LLM generated answers:
    1. Scans for hallucinated URLs not present in the retrieved source context.
    2. Scrubs unverified links to prevent phishing or broken link hallucinations.
    3. Builds deduplicated, structured Citation objects from retrieved chunks.
    """

    URL_REGEX = re.compile(r"https?://[^\s)\]>\"]+")

    @classmethod
    def _extract_urls(cls, text: str) -> List[str]:
        """Extract all HTTP/HTTPS URLs from text."""
        return cls.URL_REGEX.findall(text)

    @classmethod
    def validate_urls(
        cls, answer: str, context_chunks: List[RetrievalResult]
    ) -> Tuple[str, List[str]]:
        """
        Detect and sanitize URLs in the LLM answer.
        If a URL appears in the answer but NOT in any context chunk, replace it.
        """
        combined_context = " ".join(c.text for c in context_chunks)
        answer_urls = cls._extract_urls(answer)
        warnings = []
        cleaned_answer = answer

        for url in answer_urls:
            if url not in combined_context:
                warning_msg = f"Scrubbed hallucinated URL: {url}"
                logger.warning(warning_msg)
                warnings.append(warning_msg)
                cleaned_answer = cleaned_answer.replace(
                    url, "[URL not in source documentation]"
                )

        return cleaned_answer, warnings

    @classmethod
    def build_citations(cls, chunks: List[RetrievalResult]) -> List[Citation]:
        """
        Construct structured, deduplicated Citation objects from retrieved chunks.
        """
        citations: List[Citation] = []
        seen: Set[Tuple[str, int]] = set()

        for chunk in chunks:
            key = (chunk.doc_name, chunk.page_num)
            if key in seen:
                continue
            seen.add(key)

            citation_str = (
                f"[{chunk.doc_name} ({chunk.doc_code} v{chunk.doc_version}), "
                f"Page {chunk.page_num}]"
            )
            citations.append(
                Citation(
                    doc_name=chunk.doc_name,
                    doc_filename=chunk.doc_filename,
                    doc_code=chunk.doc_code,
                    doc_version=chunk.doc_version,
                    page_num=chunk.page_num,
                    section_heading=chunk.section_heading,
                    chunk_type=chunk.chunk_type,
                    citation_string=citation_str,
                )
            )

        return citations

    @classmethod
    def validate(
        cls, answer: str, context_chunks: List[RetrievalResult]
    ) -> Tuple[str, List[Citation], List[str]]:
        """
        Run full validation pipeline on the generated response.

        Returns:
            Tuple of (cleaned_answer, citations_list, warnings_list)
        """
        cleaned_answer, warnings = cls.validate_urls(answer, context_chunks)
        citations = cls.build_citations(context_chunks)
        return cleaned_answer, citations, warnings
