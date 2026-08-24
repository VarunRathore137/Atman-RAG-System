import re
from typing import List, Optional

from config import cfg
from src.ingestion.models import ExtractedTable
from src.chunking.models import EnrichedChunk
from src.chunking.enricher import MetadataEnricher


class TextSplitter:
    """
    Recursive character text splitter with configurable size and overlap.
    Tries natural separators (paragraph > line > sentence > word) before
    hard-cutting, ensuring chunks never start or end mid-word.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or cfg.TEXT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or cfg.TEXT_CHUNK_OVERLAP

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using the best available separator."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                tail = text[start:].strip()
                if tail:
                    chunks.append(tail)
                break

            # Scan backwards from end to find the best natural split point.
            # The split point must be at least 1/3 into the chunk to avoid
            # producing tiny leading fragments.
            split_pos = -1
            window = text[start:end]
            for sep in self.SEPARATORS:
                if not sep:
                    continue
                pos = window.rfind(sep)
                if pos >= self.chunk_size // 3:
                    split_pos = start + pos + len(sep)
                    break

            if split_pos <= start:
                split_pos = end

            chunk = text[start:split_pos].strip()
            if len(chunk) >= 15:
                chunks.append(chunk)

            # Next window starts (chunk_size - overlap) forward
            start = max(start + 1, split_pos - self.chunk_overlap)

        return chunks


class AtomicTableChunker:
    """
    Wraps a single ExtractedTable as one indivisible EnrichedChunk.
    Tables are NEVER split — a fragmented Markdown table has no meaning
    without its header row and is useless for structured Q&A retrieval.
    """

    @classmethod
    def chunk_table(
        cls,
        table: ExtractedTable,
        doc_name: str,
        doc_filename: str,
        doc_code: str,
        doc_version: str,
        doc_type: str = "data_tables",
    ) -> EnrichedChunk:
        """
        Convert an ExtractedTable to an EnrichedChunk.
        Prepends a provenance header to the Markdown so the embedding
        captures the document context alongside the table content.
        """
        provenance = (
            f"[DOCUMENT: {doc_name} | DOC_CODE: {doc_code} "
            f"| PAGE: {table.page_num} | TABLE: {table.table_id}]\n"
        )
        full_text = provenance + table.markdown_content

        if table.raw_headers:
            heading = "Table: " + ", ".join(table.raw_headers[:3])
        else:
            heading = f"Table {table.table_id}"

        return MetadataEnricher.enrich(
            chunk_id=table.table_id,
            text=full_text,
            doc_name=doc_name,
            doc_filename=doc_filename,
            doc_code=doc_code,
            doc_version=doc_version,
            page_num=table.page_num,
            chunk_type="table",
            doc_type=doc_type,
            has_table=True,
            has_code=False,
            section_heading=heading,
        )


class CodeBlockChunker:
    """
    Splits API reference pages by HTTP endpoint or section headers,
    keeping each endpoint specification as a cohesive chunk.
    Falls back to TextSplitter if no endpoint boundaries are found.
    """

    # Split before major section or endpoint headers (e.g. "1. Authentication", "3.1 Upload a File")
    ENDPOINT_PATTERN = re.compile(
        r"\n(?=[0-9]+\.[0-9]*\s+[A-Z]|###\s+)",
        re.IGNORECASE,
    )

    def __init__(self, chunk_size: Optional[int] = None):
        self.chunk_size = chunk_size or cfg.CODE_CHUNK_SIZE

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # Split on section/endpoint headers first
        raw_sections = self.ENDPOINT_PATTERN.split(text)
        if len(raw_sections) > 1:
            chunks: List[str] = []
            sub = TextSplitter(chunk_size=self.chunk_size, chunk_overlap=100)
            current_buffer = ""
            for s in raw_sections:
                s = s.strip()
                if not s:
                    continue
                # If a section is very short (e.g. just a heading like '3. Endpoints'), buffer it with next
                if len(s) < 50:
                    current_buffer = s + "\n" if not current_buffer else current_buffer + s + "\n"
                    continue
                full_s = (current_buffer + s).strip() if current_buffer else s
                current_buffer = ""
                if len(full_s) <= self.chunk_size:
                    chunks.append(full_s)
                else:
                    chunks.extend(sub.split_text(full_s))
            if current_buffer:
                if chunks:
                    chunks[-1] = chunks[-1] + "\n" + current_buffer.strip()
                else:
                    chunks.append(current_buffer.strip())
            return [c for c in chunks if c.strip()]

        if len(text) <= self.chunk_size:
            return [text]

        # No endpoint markers — use generic splitter
        return TextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=100
        ).split_text(text)


class QABoundarySplitter:
    """
    Splits FAQ / support documents on question boundaries so that each
    Q + A pair is stored as an atomic, self-contained chunk.
    A user query like 'how to reset password' should retrieve the complete
    Q+A pair, not just the question line or just the answer.
    """

    # Matches lines starting with Q:, Q., Q-, Question 1:, etc.
    QA_PATTERN = re.compile(
        r"(?=(?:^|\n)Q(?:uestion)?\s*[0-9]*\s*[:\.\-]\s*)",
        re.IGNORECASE | re.MULTILINE,
    )

    @classmethod
    def split(cls, text: str) -> List[str]:
        """Split FAQ text into atomic Q+A chunks."""
        text = text.strip()
        if not text:
            return []

        pieces = cls.QA_PATTERN.split(text)
        qa_chunks = [p.strip() for p in pieces if p.strip() and len(p.strip()) >= 20]

        # Fallback: no Q: markers found — use standard paragraph splitting
        if not qa_chunks:
            return TextSplitter(chunk_size=600, chunk_overlap=100).split_text(text)

        return qa_chunks
