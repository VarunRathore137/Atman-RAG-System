from typing import Any, Dict, List, Optional
import streamlit as st

from config import cfg
from src.generation.models import Citation, RAGResponse
from src.retrieval.models import RetrievalResult, RetrievalResponse


def render_confidence_badge(badge: str, score: float) -> str:
    """
    Generate a modern, accessible styled badge for confidence level.
    """
    badge_upper = (badge or "UNKNOWN").upper()
    percentage = f"{score * 100:.1f}%"

    if badge_upper == "HIGH":
        bg_color = "#d4edda"
        text_color = "#155724"
        border_color = "#c3e6cb"
        icon = "🟢"
        label = "HIGH CONFIDENCE"
    elif badge_upper == "MEDIUM":
        bg_color = "#fff3cd"
        text_color = "#856404"
        border_color = "#ffeeba"
        icon = "🟡"
        label = "MEDIUM CONFIDENCE"
    elif badge_upper == "LOW":
        bg_color = "#ffeeba"
        text_color = "#a71d2a"
        border_color = "#f5c6cb"
        icon = "🟠"
        label = "LOW CONFIDENCE"
    else:  # ABSTAINED or UNKNOWN
        bg_color = "#f8d7da"
        text_color = "#721c24"
        border_color = "#f5c6cb"
        icon = "🔴"
        label = "ABSTAINED (OUT-OF-DOMAIN)"

    badge_html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: {bg_color};
        color: {text_color};
        border: 1px solid {border_color};
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 10px;
        letter-spacing: 0.02em;
    ">
        <span>{icon}</span>
        <span>{label}</span>
        <span style="opacity: 0.8; font-size: 0.78rem; font-weight: 500;">({percentage})</span>
    </div>
    """
    return badge_html


def render_source_citations(
    citations: List[Citation],
    retrieval_response: Optional[RetrievalResponse] = None,
):
    """
    Render expandable provenance cards for retrieved source chunks.
    """
    if not citations and (not retrieval_response or not retrieval_response.results):
        return

    with st.expander("📚 Document Sources & Provenance Citations", expanded=False):
        # High-level citation chips
        st.markdown("**Referenced Source Documents:**")
        citation_tags = " ".join([f"`{c.citation_string}`" for c in citations])
        if citation_tags:
            st.markdown(citation_tags)

        st.markdown("---")
        st.markdown("**Retrieved Context Excerpts (Ranked by Two-Stage Pipeline):**")

        results: List[RetrievalResult] = (
            retrieval_response.results if retrieval_response else []
        )

        for i, chunk in enumerate(results, 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(
                        f"**Rank #{i}:** {chunk.doc_name} (`{chunk.doc_code} v{chunk.doc_version}`)"
                    )
                with col2:
                    st.caption(f"📄 Page {chunk.page_num} | 🏷️ {chunk.chunk_type.upper()}")
                with col3:
                    st.caption(f"🎯 Score: `{chunk.composite_score:.3f}`")

                st.caption(f"**Section Heading:** *{chunk.section_heading}*")
                
                # Render table or formatted text
                if chunk.chunk_type == "table":
                    st.markdown(chunk.text)
                else:
                    st.info(chunk.text)
                st.markdown("---")


def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar controls for provider selection, document filters, and search parameters.
    """
    with st.sidebar:
        st.markdown("### 🛡️ Atman RAG Controls")
        st.caption("Enterprise Two-Stage Q&A System")
        st.markdown("---")

        st.subheader("🔍 Scope & Retrieval")
        doc_options = [
            "All Documents",
            "Pricing_and_SLA",
            "FAQ_Support",
            "Product_Manual",
            "API_Reference",
            "Employee_Handbook",
            "Security_Policy",
            "Onboarding_Guide",
        ]
        selected_doc = st.selectbox(
            "Document Scope Filter:",
            options=doc_options,
            index=0,
            help="Filter retrieval exclusively to a specific document or query the entire corpus.",
        )
        doc_filter = None if selected_doc == "All Documents" else selected_doc

        top_k = st.slider(
            "Retrieval Candidates (Top-K):",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of candidate excerpts retrieved by ChromaDB Stage 1 and Cross-Encoder Stage 2.",
        )

        st.markdown("---")
        st.subheader("🤖 LLM Inference Provider")
        
        provider_options = ["Groq Cloud (Llama 3.3 70B)", "Ollama (Offline)", "Mock (Deterministic Testing)"]
        default_idx = 0 if cfg.GROQ_API_KEY else 2
        provider_choice = st.radio(
            "Inference Engine:",
            options=provider_options,
            index=default_idx,
            help="Select live Groq Cloud API, local offline Ollama, or deterministic mock test mode.",
        )

        provider_map = {
            "Groq Cloud (Llama 3.3 70B)": "groq",
            "Ollama (Offline)": "ollama",
            "Mock (Deterministic Testing)": "mock",
        }
        chosen_provider = provider_map[provider_choice]

        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=cfg.LLM_TEMPERATURE,
            step=0.1,
            help="0.0 provides strict, deterministic factual responses.",
        )

        st.markdown("---")
        st.subheader("🛡️ Active Guardrail Config")
        st.caption(f"• Layer 1 Threshold: `max_score >= {cfg.CONFIDENCE_THRESHOLD}`")
        st.caption(f"• Scoring Formula: `{cfg.VECTOR_WEIGHT}*Vector + {cfg.RERANK_WEIGHT}*Rerank`")
        st.caption("• Layer 2 URL Scrubber: Active")

        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        return {
            "doc_filter": doc_filter,
            "top_k": top_k,
            "provider": chosen_provider,
            "temperature": temperature,
        }


def init_session_state():
    """Initialize Streamlit session state for conversation history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
