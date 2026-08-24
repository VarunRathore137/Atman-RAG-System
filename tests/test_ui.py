"""
Phase 6 UI unit and component tests:
Tests confidence badge HTML rendering, source citation formatting, and app module integrity.
"""

import pytest

from src.ui.components import render_confidence_badge, render_source_citations
from src.generation.models import Citation, RAGResponse
from src.retrieval.models import RetrievalResult, RetrievalResponse


def test_confidence_badge_html_rendering():
    """
    Test confidence badge generator produces styled HTML for all 4 states.
    """
    high_html = render_confidence_badge("HIGH", 0.852)
    assert "HIGH CONFIDENCE" in high_html
    assert "85.2%" in high_html
    assert "#d4edda" in high_html  # Green background

    med_html = render_confidence_badge("MEDIUM", 0.62)
    assert "MEDIUM CONFIDENCE" in med_html
    assert "62.0%" in med_html
    assert "#fff3cd" in med_html  # Yellow background

    low_html = render_confidence_badge("LOW", 0.45)
    assert "LOW CONFIDENCE" in low_html
    assert "45.0%" in low_html

    abstained_html = render_confidence_badge("ABSTAINED", 0.035)
    assert "ABSTAINED" in abstained_html
    assert "#f8d7da" in abstained_html  # Red background


def test_source_citation_models_and_formatting():
    """
    Test citation data model structures used by Streamlit UI.
    """
    c = Citation(
        doc_name="Pricing_and_SLA",
        doc_filename="Pricing_and_SLA.pdf",
        doc_code="PRC-SLA-021",
        doc_version="3.2",
        page_num=2,
        section_heading="Pricing Table",
        chunk_type="table",
        citation_string="[Pricing_and_SLA (PRC-SLA-021 v3.2), Page 2]",
    )

    assert c.doc_code == "PRC-SLA-021"
    assert c.page_num == 2
    assert "Page 2" in c.citation_string


def test_app_module_import_and_structure():
    """
    Verify src.ui.app and src.ui.components import without syntax or import errors.
    """
    import src.ui.components as components
    import src.ui.app as app

    assert hasattr(components, "render_confidence_badge")
    assert hasattr(components, "render_source_citations")
    assert hasattr(components, "render_sidebar")
    assert hasattr(app, "main")
    assert hasattr(app, "get_cached_pipeline")
