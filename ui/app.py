# =============================================================================
# ui/app.py
#
# DiaVeritas Streamlit application.
#
# Design philosophy: "Don't make the LLM the star. Make the evidence pipeline the star."
#
# Two views:
#   1. Answer View — clean, academic-style result display
#   2. Research/Debug View — full reasoning trail for the NLP pipeline
#
# UI aesthetic: clean white/light neutral, restrained accent, professional
# typography, whitespace-heavy, academic research tool.
# =============================================================================

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

from src.config import settings


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DiaVeritas — T2D Evidence Analysis",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------------------------------------------------------
# Pipeline initialization (cached per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading pipeline components...")
def load_pipeline():
    """
    Load all pipeline components.
    Returns (synthesizer, error_message).
    If components fail to load, returns (None, error_str).
    """
    try:
        from src.retrieval.embedder import Embedder
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.bm25_index import BM25Index
        from src.generation.synthesizer import Synthesizer

        embedder = Embedder()
        vector_store = VectorStore()

        # Load BM25 index
        bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
        if not bm25_path.exists():
            return None, (
                "BM25 index not found. Please run: `py scripts/build_index.py --step index`"
            )
        bm25 = BM25Index.load(bm25_path)

        if vector_store.count() == 0:
            return None, (
                "Vector store is empty. Please run: `py scripts/build_index.py`"
            )

        # LLM (optional — system degrades gracefully without it)
        llm = None
        try:
            from src.generation.llm_client import LLMClient
            candidate_llm = LLMClient()
            candidate_llm._load()
            llm = candidate_llm
        except Exception as e:
            logger.warning(f"LLM not available: {e}. Running without LLM synthesis.")
            llm = None

        synthesizer = Synthesizer(
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=bm25,
            llm_client=llm,
        )
        return synthesizer, None

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="dv-header">
        <div class="dv-wordmark">DiaVeritas</div>
        <div class="dv-tagline">Evidence analysis for Type 2 Diabetes treatment literature</div>
        <div class="dv-disclaimer">
            Research prototype · Not medical advice · Not a clinical decision system
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Processing indicator
# ---------------------------------------------------------------------------

def render_pipeline_steps(active_step: int):
    steps = [
        "Retrieving evidence",
        "Analyzing claims",
        "Checking NLI relationships",
        "Comparing contexts",
        "Synthesizing evidence",
    ]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < active_step:
                st.markdown(f'<div class="step-done">✓ {step}</div>', unsafe_allow_html=True)
            elif i == active_step:
                st.markdown(f'<div class="step-active">⟳ {step}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="step-pending">○ {step}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Answer view (clean UI)
# ---------------------------------------------------------------------------

def render_answer_view(result):
    """Render the clean evidence-grounded answer view."""

    # Evidence Status
    status = result.status
    status_colors = {
        "SUPPORTED": "status-supported",
        "REFUTED": "status-refuted",
        "INCONCLUSIVE": "status-inconclusive",
    }
    status_class = status_colors.get(status, "status-inconclusive")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="status-badge {status_class}">
            EVIDENCE STATUS: {status}
        </div>
        """, unsafe_allow_html=True)
    with col2:
        status_dict = result.status_result_dict
        conf_label = status_dict.get("confidence_label", "Low")
        conf_score = status_dict.get("confidence_score", 0.0)
        st.markdown(f"""
        <div class="confidence-box">
            <div class="conf-label">Prototype Confidence</div>
            <div class="conf-value">{conf_label} ({conf_score:.0%})</div>
            <div class="conf-note">Not clinically calibrated</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        ev_sum = result.evidence_summary_dict
        st.markdown(f"""
        <div class="evidence-counts">
            <span class="ev-sup">● {ev_sum.get('n_supporting', 0)} supporting</span><br>
            <span class="ev-con">● {ev_sum.get('n_contradicting', 0)} contradicting</span><br>
            <span class="ev-ctx">● {ev_sum.get('n_contextual', 0)} contextual</span><br>
            <span class="ev-neu">● {ev_sum.get('n_neutral', 0)} neutral</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Answer
    st.markdown('<div class="section-label">SYNTHESIZED ANSWER</div>', unsafe_allow_html=True)
    if result.answer:
        st.markdown(result.answer)
    if result.answer_error:
        st.error(f"LLM error: {result.answer_error}")

    st.markdown("---")

    # Evidence panels
    _render_evidence_panels(result)

    # Rationale
    if status_dict.get("rationale"):
        st.markdown('<div class="section-label">EVIDENCE STATUS RATIONALE</div>', unsafe_allow_html=True)
        st.info(status_dict["rationale"])

    # Latency
    st.caption(
        f"Pipeline: {result.pipeline_mode} · "
        f"Latency: {result.latency_seconds}s · "
        f"Models: {settings.embedding_model} | {settings.nli_model}"
    )


def _render_evidence_panels(result):
    """Render expandable evidence panels grouped by relationship."""
    ev = result.evidence_summary_dict

    tabs = st.tabs(["Supporting Evidence", "Contradicting Evidence", "Contextual Differences", "Sources"])

    with tabs[0]:
        items = ev.get("supporting", [])
        if not items:
            st.write("No supporting evidence retrieved.")
        for item in items:
            _render_evidence_card(item, "supporting")

    with tabs[1]:
        items = ev.get("contradicting", [])
        if not items:
            st.write("No contradicting evidence retrieved.")
        for item in items:
            _render_evidence_card(item, "contradicting")

    with tabs[2]:
        items = ev.get("contextual", [])
        if not items:
            st.write("No contextual differences detected.")
        for item in items:
            _render_evidence_card(item, "contextual")

    with tabs[3]:
        _render_sources(result)


def _render_evidence_card(item: dict, card_type: str):
    title = item.get("title", "Not reported")
    year = item.get("year", "?")
    journal = item.get("journal", "Not reported")
    section = item.get("section", "Unknown")
    study_type = item.get("study_type", "Not reported")
    text = item.get("text", "")[:500]
    claim = item.get("claim", {})
    nli = item.get("nli", {})
    ctx = item.get("context_analysis")

    header = f"**{title[:80]}** ({year}, {journal})"
    with st.expander(header, expanded=False):
        cols = st.columns(3)
        cols[0].markdown(f"**Section:** {section}")
        cols[1].markdown(f"**Study type:** {study_type}")
        cols[2].markdown(f"**NLI:** {nli.get('label', '?')} ({nli.get('confidence', 0):.2f})")

        st.markdown("**Extracted claim:**")
        claim_parts = []
        if claim.get("intervention") != "Not reported":
            claim_parts.append(f"*Intervention:* {claim['intervention']}")
        if claim.get("outcome") != "Not reported":
            claim_parts.append(f"*Outcome:* {claim['outcome']}")
        if claim.get("direction") not in ("Not reported", "Unclear"):
            claim_parts.append(f"*Direction:* {claim['direction']}")
        if claim_parts:
            st.markdown(" · ".join(claim_parts))
        else:
            st.markdown("*(claim extraction limited)*")

        st.markdown("**Evidence passage:**")
        st.markdown(f"> {text}...")

        if ctx and ctx.get("differences"):
            st.markdown("**Contextual differences identified:**")
            for diff in ctx["differences"][:3]:
                st.markdown(f"  - [{diff['dimension'].upper()}] {diff['description']}")


def _render_sources(result):
    """Render the consolidated source bibliography."""
    seen_papers = {}
    ev = result.evidence_summary_dict

    all_items = (
        ev.get("supporting", []) +
        ev.get("contradicting", []) +
        ev.get("contextual", []) +
        ev.get("neutral", [])
    )

    for item in all_items:
        pid = item.get("paper_id", "?")
        if pid not in seen_papers:
            seen_papers[pid] = item

    if not seen_papers:
        st.write("No sources available.")
        return

    for pid, item in seen_papers.items():
        authors = item.get("authors", ["Not reported"])
        year = item.get("year", "?")
        title = item.get("title", "Not reported")
        journal = item.get("journal", "Not reported")
        study_type = item.get("study_type", "Not reported")
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        st.markdown(
            f"- **{author_str} ({year}).** {title}. *{journal}*. "
            f"[{study_type}] [PMID: {pid}]"
        )


# ---------------------------------------------------------------------------
# Research / Debug view
# ---------------------------------------------------------------------------

def render_debug_view(result):
    """Render the full reasoning trail for the research/debug view."""
    st.markdown('<div class="debug-header">RESEARCH / DEBUG VIEW — Full Reasoning Trail</div>',
                unsafe_allow_html=True)
    st.caption(
        "This view exposes the complete NLP pipeline. "
        "It is intended for research inspection and viva demonstration."
    )

    # 1. Query
    with st.expander("① QUESTION + NORMALIZED QUERY", expanded=True):
        st.markdown(f"**Question:** {result.question}")
        nq = result.normalized_query
        if nq:
            cols = st.columns(3)
            cols[0].markdown(f"**Interventions:**\n{', '.join(nq.get('interventions', []) or ['—'])}")
            cols[1].markdown(f"**Outcomes:**\n{', '.join(nq.get('outcomes', []) or ['—'])}")
            cols[2].markdown(f"**Population:**\n{', '.join(nq.get('population', []) or ['—'])}")
            st.markdown(f"**Normalized text:** `{nq.get('normalized_text', '—')}`")

    # 2. Retrieval candidates
    with st.expander(f"② RETRIEVED CANDIDATES ({len(result.candidates)} total)", expanded=False):
        if result.candidates:
            st.markdown("Top 10 RRF candidates (before reranking):")
            for item in result.candidates[:10]:
                cols = st.columns(4)
                cols[0].markdown(f"**Rank {item.get('rank', '?')}**")
                cols[1].metric("Dense", f"{item.get('dense_score', 0):.3f}")
                cols[2].metric("BM25", f"{item.get('bm25_score', 0):.3f}")
                cols[3].metric("RRF", f"{item.get('rrf_score', 0):.4f}")
                st.markdown(f"> {item.get('text', '')[:200]}...")
                st.divider()
        else:
            st.warning("No candidates retrieved.")

    # 3. Reranked
    with st.expander(f"③ RERANKED TOP-{len(result.reranked)} EVIDENCE", expanded=False):
        for item in result.reranked:
            cols = st.columns(3)
            cols[0].markdown(f"**Rank {item.get('reranker_rank', '?')}**")
            cols[1].metric("Reranker score", f"{item.get('reranker_score', 0):.3f}")
            chunk_obj = item.get('chunk')
            section_label = getattr(chunk_obj, 'section', '?') if chunk_obj is not None else '?'
            cols[2].markdown(f"Section: {section_label}")
            st.markdown(f"> {item.get('text', '')[:300]}...")
            st.divider()

    # 4. Claims
    with st.expander(f"④ EXTRACTED CLAIMS ({len(result.claims)})", expanded=False):
        for claim in result.claims:
            with st.container():
                st.markdown(f"**Chunk:** `{claim.chunk_id}`")
                cols = st.columns(4)
                cols[0].markdown(f"**Intervention:** {claim.intervention}")
                cols[1].markdown(f"**Outcome:** {claim.outcome}")
                cols[2].markdown(f"**Direction:** {claim.direction}")
                cols[3].markdown(f"**Duration:** {claim.duration}")
                st.markdown(f"*Context:* {claim.study_context}")
                st.divider()

    # 5. NLI
    with st.expander("⑤ NLI RESULTS", expanded=False):
        nli_sum = result.nli_summary
        cols = st.columns(4)
        counts = nli_sum.get("counts", {})
        cols[0].metric("Entailments", counts.get("ENTAILMENT", 0))
        cols[1].metric("Contradictions", counts.get("CONTRADICTION", 0))
        cols[2].metric("Neutral", counts.get("NEUTRAL", 0))
        cols[3].metric("Avg confidence", f"{nli_sum.get('avg_confidence', 0):.3f}")

    # 6. Context analysis
    ev = result.evidence_summary_dict
    contextual_items = ev.get("contextual", [])
    with st.expander(f"⑥ CONTEXTUAL CONTRADICTION ANALYSIS ({len(contextual_items)} items)", expanded=False):
        for item in contextual_items:
            ctx = item.get("context_analysis")
            if ctx:
                st.markdown(f"**Chunk:** {item.get('chunk_id', '?')}")
                st.markdown(f"NLI: {ctx.get('nli_label')} ({ctx.get('nli_confidence', 0):.2f}) → Conflict type: **{ctx.get('conflict_type')}**")
                if ctx.get("differences"):
                    for diff in ctx["differences"]:
                        st.markdown(f"  - [{diff['dimension'].upper()}] {diff['description']}")
                st.divider()

    # 7. Evidence relationships
    with st.expander("⑦ EVIDENCE RELATIONSHIP SUMMARY", expanded=True):
        ev = result.evidence_summary_dict
        cols = st.columns(4)
        cols[0].metric("Supporting", ev.get("n_supporting", 0))
        cols[1].metric("Contradicting", ev.get("n_contradicting", 0))
        cols[2].metric("Contextual Diff", ev.get("n_contextual", 0))
        cols[3].metric("Neutral", ev.get("n_neutral", 0))

    # 8. Final status
    with st.expander("⑧ EVIDENCE STATUS", expanded=True):
        s = result.status_result_dict
        st.markdown(f"### {s.get('status', 'INCONCLUSIVE')}")
        cols = st.columns(3)
        cols[0].metric("Entailment ratio", f"{s.get('entailment_ratio', 0):.1%}")
        cols[1].metric("Contradiction ratio", f"{s.get('contradiction_ratio', 0):.1%}")
        cols[2].metric("Confidence", f"{s.get('confidence_label', '?')} ({s.get('confidence_score', 0):.2f})")
        st.markdown(f"**Rationale:** {s.get('rationale', '—')}")

    # 9. LLM synthesis prompt preview
    with st.expander("⑨ LLM SYNTHESIS (Answer Preview)", expanded=False):
        st.markdown(result.answer or "*No LLM answer generated.*")
        if result.answer_error:
            st.error(f"Error: {result.answer_error}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    render_header()

    # View selector
    view = st.radio(
        "View:",
        ["Answer View", "Research / Debug View"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Query input
    query = st.text_area(
        "Ask a question about Type 2 Diabetes treatment evidence...",
        placeholder="e.g., Does metformin reduce cardiovascular risk in patients with Type 2 Diabetes?",
        height=80,
        key="query_input",
        label_visibility="collapsed",
    )

    col_submit, col_baseline, col_clear = st.columns([1, 1, 3])
    run_full = col_submit.button("🔍 Analyze Evidence", type="primary", use_container_width=True)
    run_baseline = col_baseline.button("⚡ Baseline RAG", use_container_width=True)

    st.markdown("---")

    # Load pipeline
    synthesizer, pipeline_error = load_pipeline()

    if pipeline_error:
        st.error(f"**Pipeline not ready:** {pipeline_error}")
        st.info(
            "To build the index:\n"
            "```bash\n"
            "py scripts/build_index.py --step all --max-papers 30\n"
            "```"
        )
        return

    # Run pipeline
    if (run_full or run_baseline) and query.strip():
        mode = "baseline" if run_baseline else "diaveritias"

        # Processing indicator
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown("**Processing...**")
            for step_i in range(5):
                render_pipeline_steps(step_i)
                time.sleep(0.1)

        with st.spinner("Running evidence analysis pipeline..."):
            result = synthesizer.run(query.strip(), mode=mode)

        progress_placeholder.empty()

        # Store in session state
        st.session_state["last_result"] = result

    # Render result
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        if view == "Answer View":
            render_answer_view(result)
        else:
            render_debug_view(result)

    # Empty state
    elif not query.strip():
        st.markdown("""
        <div class="empty-state">
            <p>Enter a question above to analyze Type 2 Diabetes treatment evidence.</p>
            <p style="font-size: 0.9rem; color: #888;">
                Example: "Does SGLT2 inhibitor treatment reduce heart failure hospitalization in T2D?"
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
