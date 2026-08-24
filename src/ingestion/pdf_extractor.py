import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
import pdfplumber
import fitz  # PyMuPDF

from config.logging_config import logger
from src.ingestion.models import PageDocument, ExtractedTable
from src.ingestion.table_processor import TableProcessor

class PDFExtractor:
    """
    High-fidelity multi-modal PDF extraction engine.
    Uses pdfplumber as primary extractor to preserve tabular cell matrices as Markdown,
    with PyMuPDF (fitz) as a robust high-speed fallback.
    """
    DOC_CODE_PATTERN = re.compile(
        r"Document Code:\s*([A-Z0-9\-]+)\s*\|?\s*Version\s*([\d\.]+)?",
        re.IGNORECASE
    )

    def __init__(self, doc_dir: Optional[Path] = None):
        self.doc_dir = Path(doc_dir) if doc_dir else None

    def extract_doc_metadata(self, text: str) -> Tuple[str, str]:
        match = self.DOC_CODE_PATTERN.search(text)
        if match:
            code = match.group(1).strip()
            version = match.group(2).strip() if match.group(2) else "3.2"
            return code, version
        return "UNKNOWN", "1.0"

    def extract_pdf(self, pdf_path: Path) -> List[PageDocument]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_filename = pdf_path.name
        doc_name = pdf_path.stem
        page_documents: List[PageDocument] = []
        doc_code, doc_version = "UNKNOWN", "1.0"

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    extracted_text = page.extract_text() or ""
                    
                    if page_num == 1:
                        doc_code, doc_version = self.extract_doc_metadata(extracted_text)

                    raw_tables = page.extract_tables() or []
                    tables: List[ExtractedTable] = []

                    for t_idx, raw_table in enumerate(raw_tables):
                        t_id = f"{doc_name}__p{page_num:03d}__t{t_idx+1:02d}"
                        extracted_table = TableProcessor.to_markdown(
                            raw_table, table_id=t_id, page_num=page_num
                        )
                        if extracted_table:
                            tables.append(extracted_table)

                    page_doc = PageDocument(
                        doc_name=doc_name,
                        doc_filename=doc_filename,
                        doc_code=doc_code,
                        doc_version=doc_version,
                        page_num=page_num,
                        text=extracted_text.strip(),
                        tables=tables,
                        has_tables=len(tables) > 0,
                        extractor_used="pdfplumber",
                        metadata={
                            "total_pages": total_pages,
                            "table_count": len(tables),
                            "char_length": len(extracted_text.strip())
                        }
                    )
                    page_documents.append(page_doc)

            logger.info(
                f"Extracted '{doc_filename}' via pdfplumber: {len(page_documents)} pages, "
                f"{sum(len(p.tables) for p in page_documents)} tables."
            )

        except Exception as e:
            logger.warning(f"pdfplumber encountered issue on '{doc_filename}': {e}. Falling back to PyMuPDF.")
            page_documents = self._fallback_pymupdf(pdf_path, doc_name, doc_filename)

        return page_documents

    def _fallback_pymupdf(self, pdf_path: Path, doc_name: str, doc_filename: str) -> List[PageDocument]:
        page_documents: List[PageDocument] = []
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc_code, doc_version = "UNKNOWN", "1.0"

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text() or ""
            
            if page_num == 1:
                doc_code, doc_version = self.extract_doc_metadata(text)

            page_doc = PageDocument(
                doc_name=doc_name,
                doc_filename=doc_filename,
                doc_code=doc_code,
                doc_version=doc_version,
                page_num=page_num,
                text=text.strip(),
                tables=[],
                has_tables=False,
                extractor_used="pymupdf_fallback",
                metadata={
                    "total_pages": total_pages,
                    "table_count": 0,
                    "char_length": len(text.strip())
                }
            )
            page_documents.append(page_doc)

        doc.close()
        logger.info(f"Extracted '{doc_filename}' via PyMuPDF fallback: {len(page_documents)} pages.")
        return page_documents

    def extract_all(self, directory: Optional[Path] = None) -> List[PageDocument]:
        target_dir = Path(directory) if directory else (self.doc_dir or Path("files"))
        if not target_dir.exists():
            raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

        all_pages: List[PageDocument] = []
        pdf_files = sorted(target_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in: {target_dir}")
            return []

        for pdf_file in pdf_files:
            all_pages.extend(self.extract_pdf(pdf_file))

        logger.info(f"Completed batch extraction: {len(pdf_files)} PDFs -> {len(all_pages)} total pages.")
        return all_pages
