from typing import List, Optional

from config.logging_config import logger
from src.ingestion.models import PageDocument
from src.chunking.models import EnrichedChunk, DOC_TYPE_MAP
from src.chunking.enricher import MetadataEnricher
from src.chunking.splitters import (
    TextSplitter,
    AtomicTableChunker,
    CodeBlockChunker,
    QABoundarySplitter,
)


class SemanticChunker:
    """
    Orchestrates document-type aware chunking across the heterogeneous PDF corpus.

    Routing logic (per DOC_TYPE_MAP):
        "qa_pairs"       -> QABoundarySplitter  (FAQ_Support)
        "technical_api"  -> CodeBlockChunker    (API_Reference)
        everything else  -> TextSplitter        (Manual, Handbook, Security, etc.)

    Tables are ALWAYS processed atomically first, regardless of doc_type.
    """

    def __init__(
        self,
        text_chunk_size: Optional[int] = None,
        text_chunk_overlap: Optional[int] = None,
    ):
        self.text_splitter = TextSplitter(
            chunk_size=text_chunk_size, chunk_overlap=text_chunk_overlap
        )
        self.code_splitter = CodeBlockChunker()

    def chunk_page(
        self, page: PageDocument, start_idx: int = 1
    ) -> List[EnrichedChunk]:
        """
        Convert a single PageDocument into a list of EnrichedChunks.
        Tables are chunked atomically first; then text is routed by doc_type.
        """
        doc_type = DOC_TYPE_MAP.get(page.doc_name, "general")
        chunks: List[EnrichedChunk] = []
        text_idx = start_idx

        # Step 1: Atomic table chunks (always, regardless of doc type)
        if page.has_tables and page.tables:
            for table in page.tables:
                table_chunk = AtomicTableChunker.chunk_table(
                    table=table,
                    doc_name=page.doc_name,
                    doc_filename=page.doc_filename,
                    doc_code=page.doc_code or "UNKNOWN",
                    doc_version=page.doc_version or "1.0",
                    doc_type=doc_type,
                )
                chunks.append(table_chunk)

        # Step 2: Route text by doc_type
        page_text = page.text.strip()
        if not page_text:
            return chunks

        if doc_type == "qa_pairs":
            segments = QABoundarySplitter.split(page_text)
            chunk_type = "qa_pair"
        elif doc_type == "technical_api":
            segments = self.code_splitter.split_text(page_text)
            chunk_type = "code"
        else:
            segments = self.text_splitter.split_text(page_text)
            chunk_type = "text"

        # Step 3: Enrich each text segment
        for segment in segments:
            if not segment.strip():
                continue
            chunk_id = f"{page.doc_name}__p{page.page_num:03d}__c{text_idx:03d}"
            enriched = MetadataEnricher.enrich(
                chunk_id=chunk_id,
                text=segment,
                doc_name=page.doc_name,
                doc_filename=page.doc_filename,
                doc_code=page.doc_code or "UNKNOWN",
                doc_version=page.doc_version or "1.0",
                page_num=page.page_num,
                chunk_type=chunk_type,
                doc_type=doc_type,
                has_table=False,
            )
            chunks.append(enriched)
            text_idx += 1

        return chunks

    def chunk_all(self, pages: List[PageDocument]) -> List[EnrichedChunk]:
        """
        Chunk the entire corpus of PageDocuments into EnrichedChunks.
        Maintains per-doc text chunk counters for unique chunk IDs.
        """
        all_chunks: List[EnrichedChunk] = []
        doc_counters: dict = {}  # doc_name -> next text chunk index

        for page in pages:
            counter = doc_counters.get(page.doc_name, 1)
            page_chunks = self.chunk_page(page, start_idx=counter)

            # Advance counter only for text chunks (tables use table_id as chunk_id)
            text_count = sum(
                1 for c in page_chunks if c.chunk_type != "table"
            )
            doc_counters[page.doc_name] = counter + text_count
            all_chunks.extend(page_chunks)

        table_count = sum(1 for c in all_chunks if c.has_table)
        qa_count = sum(1 for c in all_chunks if c.chunk_type == "qa_pair")
        code_count = sum(1 for c in all_chunks if c.chunk_type == "code")
        logger.info(
            f"Chunking complete: {len(pages)} pages -> {len(all_chunks)} chunks "
            f"({table_count} tables, {qa_count} qa_pairs, {code_count} code, "
            f"{len(all_chunks) - table_count - qa_count - code_count} text)"
        )
        return all_chunks
