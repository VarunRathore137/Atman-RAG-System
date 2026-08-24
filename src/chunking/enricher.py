import re
from typing import Optional

from src.chunking.models import EnrichedChunk


class MetadataEnricher:
    """
    Extracts section headings via heuristic regex patterns and enriches
    a text segment into a fully-populated EnrichedChunk.
    """

    # Ordered list of patterns — first match wins.
    # Designed for Atman Cloud PDF structure (numbered sections, Q&A, Day markers).
    HEADING_PATTERNS = [
        # Numbered section: "1. Introduction" or "2.3 Configuration"
        re.compile(
            r"^(?:Section\s+)?([0-9]+(?:\.[0-9]+)*\s*[:\.\-]?\s+[^\n]{3,60})",
            re.MULTILINE | re.IGNORECASE,
        ),
        # Q&A question line: "Q: How do I...?" or "Q. What is...?"
        re.compile(r"^(Q[:\.\-]\s*[^\n\?]{3,80}\??)", re.MULTILINE | re.IGNORECASE),
        # Onboarding guide: "Day 1 –" or "Week 2:"
        re.compile(r"^((?:Day|Week)\s+[0-9]+[:\s\-]+[^\n]{3,60})", re.MULTILINE | re.IGNORECASE),
        # Markdown headings: "## Overview"
        re.compile(r"^(#{1,4}\s+[^\n]{3,60})", re.MULTILINE),
        # Short ALL-CAPS label: "OVERVIEW:" or "BENEFITS:"
        re.compile(r"^([A-Z][A-Z0-9\s\-_]{2,38}:)", re.MULTILINE),
    ]

    @classmethod
    def detect_heading(cls, text: str, default: str = "General") -> str:
        """Extract the most prominent section heading from the chunk text."""
        if not text:
            return default

        for pattern in cls.HEADING_PATTERNS:
            match = pattern.search(text)
            if match:
                heading = match.group(1).strip()
                # Strip leading markdown # characters
                heading = re.sub(r"^#+\s*", "", heading).strip()
                if len(heading) >= 4:
                    return heading[:100]

        # Fallback: use the first non-empty line if it looks like a title
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines and 4 <= len(lines[0]) <= 80 and not lines[0].endswith("."):
            return lines[0]

        return default

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Rough token count estimate: ~4 chars per token for English prose."""
        return max(1, len(text) // 4)

    @classmethod
    def enrich(
        cls,
        chunk_id: str,
        text: str,
        doc_name: str,
        doc_filename: str,
        doc_code: str,
        doc_version: str,
        page_num: int,
        chunk_type: str = "text",
        doc_type: str = "general",
        has_table: bool = False,
        has_code: bool = False,
        section_heading: Optional[str] = None,
    ) -> EnrichedChunk:
        """
        Factory method: create an EnrichedChunk from a raw text segment.
        Auto-detects has_code and section_heading if not explicitly provided.
        """
        cleaned = text.strip()
        heading = section_heading if section_heading else cls.detect_heading(cleaned)

        # Auto-detect code indicators if caller did not explicitly set has_code
        if not has_code:
            has_code = any(
                token in cleaned
                for token in ["```", "HTTP/1.1", "curl ", '{"', "Authorization: Bearer"]
            )

        return EnrichedChunk(
            chunk_id=chunk_id,
            doc_name=doc_name,
            doc_filename=doc_filename,
            doc_code=doc_code or "UNKNOWN",
            doc_version=doc_version or "1.0",
            page_num=page_num,
            section_heading=heading,
            chunk_type=chunk_type,
            text=cleaned,
            char_count=len(cleaned),
            token_count_est=cls.estimate_tokens(cleaned),
            has_table=has_table,
            has_code=has_code,
            doc_type=doc_type,
        )
