import pytest
from pathlib import Path
from config import cfg
from src.ingestion import PDFExtractor, PageDocument, TableProcessor

def test_pdf_extractor_all_documents():
    extractor = PDFExtractor(doc_dir=cfg.DOCS_DIR)
    pages = extractor.extract_all()
    
    assert len(pages) > 0
    doc_names = set(p.doc_name for p in pages)
    expected_docs = {
        "API_Reference",
        "Employee_Handbook",
        "FAQ_Support",
        "Onboarding_Guide",
        "Pricing_and_SLA",
        "Product_Manual",
        "Security_Policy",
    }
    assert expected_docs.issubset(doc_names)

    for page in pages:
        assert page.doc_name
        assert page.page_num >= 1
        assert len(page.text) > 0

def test_table_extraction_pricing_sla():
    extractor = PDFExtractor(doc_dir=cfg.DOCS_DIR)
    pdf = cfg.DOCS_DIR / "Pricing_and_SLA.pdf"
    pages = extractor.extract_pdf(pdf)
    
    assert len(pages) == 2
    page_2 = pages[1]
    assert page_2.has_tables
    assert len(page_2.tables) >= 1
    
    table_md = "\n".join(t.markdown_content for t in page_2.tables)
    assert "Free" in table_md and "Standard" in table_md and "Enterprise" in table_md
    assert "|" in table_md

def test_table_processor_cleaning():
    raw_grid = [
        ["Tier", "Price", "Uptime"],
        ["Standard\nPlan", "$12 / mo", "99.5%"],
        ["Enterprise", "Custom | Negotiated", "99.95%"]
    ]
    table = TableProcessor.to_markdown(raw_grid, table_id="test_table", page_num=1)
    assert table is not None
    assert table.row_count == 3
    assert table.col_count == 3
    assert "| Tier | Price | Uptime |" in table.markdown_content
    assert "Standard Plan" in table.markdown_content

def test_doc_metadata_extraction():
    extractor = PDFExtractor()
    code, version = extractor.extract_doc_metadata(
        "CloudSync Pro - User Manual\nDocument Code: PM-CSP-001 | Version 3.2 | Atman Cloud"
    )
    assert code == "PM-CSP-001"
    assert version == "3.2"
