# =============================================================================
# ui/app.py
#
# DiaVeritas Streamlit application.
#
# Design philosophy: "Don't make the LLM the star. Make the evidence pipeline the star."
#
# Clean, academic-style evidence analysis dashboard for Type 2 Diabetes literature.
# Features:
#   - Preset clinical query chips for one-click testing
#   - Clear, labeled input area
#   - Evidence Verdict & Confidence summary cards
#   - Tabbed presentation:
#       1. 📋 Clinical Synthesis (Answer, Rationale, Evidence Cards)
#       2. 🔬 Pipeline Reasoning Trail (Query PICO, Hybrid Retrieval, NLI, Contradictions)
#       3. 📚 Ingested Corpus Explorer (Search indexed PubMed papers)
# =============================================================================

from __future__ import annotations

import sys
import json
import time
import threading
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
    page_title="DiaVeritas — T2D Clinical Evidence Synthesis",
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
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------------------------------------------------------
# Preset Queries
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    {
        "label": "💊 Metformin vs Lifestyle (Prediabetes)",
        "query": "Is metformin more effective than lifestyle changes for prediabetes?",
    },
    {
        "label": "🫀 Metformin & Cardiovascular Risk",
        "query": "Does metformin reduce cardiovascular risk in patients with Type 2 Diabetes?",
    },
    {
        "label": "🩺 SGLT2i & Heart Failure",
        "query": "Are SGLT2 inhibitors effective in reducing heart failure hospitalization in T2D?",
    },
    {
        "label": "📉 GLP-1 RA & Glycemic Control",
        "query": "Do GLP-1 receptor agonists improve glycemic control and reduce HbA1c in type 2 diabetes?",
    },
]

# ---------------------------------------------------------------------------
# Pipeline initialization (cached per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load all pipeline components and pre-warm models to eliminate runtime latency.
    Returns (synthesizer, error_message).
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
            return None, "BM25 index not found. Please run: `py scripts/build_index.py --step index`"
        
        bm25 = BM25Index.load(bm25_path)

        if vector_store.count() == 0:
            return None, "Vector store is empty. Please run: `py scripts/build_index.py`"

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

        # Pre-warm all models into RAM during startup to eliminate cold-start query latency
        embedder._load()
        synthesizer._reranker._load()
        from src.nli.nli_classifier import _load_nli_model
        _load_nli_model()

        return synthesizer, None

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def get_chunk_count() -> int:
    """Get the current count of processed chunks."""
    try:
        chunks_file = Path(settings.processed_dir) / "chunks.jsonl"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
    except Exception:
        pass
    return 15513


def render_header():
    chunk_count = get_chunk_count()
    st.markdown(f"""
    <div class="dv-header">
        <div class="dv-header-top">
            <div class="dv-wordmark">DiaVeritas</div>
            <div class="dv-badge-container">
                <span class="dv-status-pill">● {chunk_count:,} PubMed Chunks</span>
                <span class="dv-status-pill">⚡ Groq Qwen-27B</span>
                <span class="dv-status-pill">🔬 DeBERTa-v3 NLI</span>
            </div>
        </div>
        <div class="dv-tagline">Evidence synthesis & conflict resolution for Type 2 Diabetes clinical literature</div>
        <div class="dv-disclaimer">
            Academic prototype for biomedical NLP evaluation · Not medical advice · Not a clinical decision support system
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Processing indicator
# ---------------------------------------------------------------------------

def render_pipeline_steps(active_step: int):
    steps = [
        "1. PICO Query Expansion",
        "2. Hybrid BM25 + Dense Retrieval",
        "3. Cross-Encoder Reranking",
        "4. DeBERTa NLI Classification",
        "5. Groq Clinical Synthesis",
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
# Status Banner
# ---------------------------------------------------------------------------

def render_verdict_banner(result):
    """Render the top summary metrics and verdict card."""
    status = result.status
    status_dict = result.status_result_dict
    ev_sum = result.evidence_summary_dict
    conf_label = status_dict.get("confidence_label", "Moderate")
    conf_score = status_dict.get("confidence_score", 0.5)

    badge_theme = {
        "SUPPORTED": {"class": "status-supported", "icon": "✓", "desc": "Evidence strongly supports the proposition"},
        "REFUTED": {"class": "status-refuted", "icon": "✗", "desc": "Evidence refutes or contradicts the proposition"},
        "INCONCLUSIVE": {"class": "status-inconclusive", "icon": "⚖", "desc": "Literature shows conflicting or insufficient evidence"},
    }.get(status, {"class": "status-inconclusive", "icon": "⚖", "desc": "Evaluation complete"})

    col1, col2, col3 = st.columns([1.5, 1.2, 1.3])

    with col1:
        st.markdown(f"""
        <div class="card-verdict {badge_theme['class']}">
            <div class="verdict-label">EVIDENCE VERDICT</div>
            <div class="verdict-title">{badge_theme['icon']} {status}</div>
            <div class="verdict-desc">{badge_theme['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card-metric">
            <div class="metric-header">CONFIDENCE CALIBRATION</div>
            <div class="metric-large-value">{conf_label} <span style="font-size: 1.1rem; color: #64748b;">({conf_score:.0%})</span></div>
            <div class="metric-progress-bar">
                <div class="metric-progress-fill" style="width: {int(conf_score * 100)}%;"></div>
            </div>
            <div class="metric-footer">Heuristic NLI agreement · Not clinically calibrated</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        n_sup = ev_sum.get('n_supporting', 0)
        n_con = ev_sum.get('n_contradicting', 0)
        n_ctx = ev_sum.get('n_contextual', 0)
        n_neu = ev_sum.get('n_neutral', 0)
        st.markdown(f"""
        <div class="card-metric">
            <div class="metric-header">EVIDENCE BREAKDOWN</div>
            <div class="evidence-grid">
                <span class="ev-chip chip-sup"><b>{n_sup}</b> Supporting</span>
                <span class="ev-chip chip-con"><b>{n_con}</b> Contradicting</span>
                <span class="ev-chip chip-ctx"><b>{n_ctx}</b> Contextual</span>
                <span class="ev-chip chip-neu"><b>{n_neu}</b> Neutral</span>
            </div>
            <div class="metric-footer">Across top reranked literature chunks</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 1: Clinical Synthesis
# ---------------------------------------------------------------------------

def render_clinical_synthesis(result):
    """Render the primary synthesized clinical answer and evidence panels."""
    # Synthesized Answer Card
    st.markdown('<div class="section-heading">Clinical Evidence Synthesis</div>', unsafe_allow_html=True)
    
    if result.answer:
        st.markdown(f"""
        <div class="synthesis-box">
            {result.answer}
        </div>
        """, unsafe_allow_html=True)
    elif result.answer_error:
        st.error(f"Synthesis error: {result.answer_error}")
    else:
        st.info("No synthesis available.")

    # Status Rationale
    status_dict = result.status_result_dict
    if status_dict.get("rationale"):
        with st.expander("📌 Methodological Rationale for Verdict", expanded=False):
            st.markdown(f"**Pipeline Verdict Explanation:**\n\n{status_dict['rationale']}")

    st.markdown('<div class="section-heading" style="margin-top: 2rem;">Retrieved Evidence Passages & Study Context</div>', unsafe_allow_html=True)
    
    ev = result.evidence_summary_dict
    sup_items = ev.get("supporting", [])
    con_items = ev.get("contradicting", [])
    ctx_items = ev.get("contextual", [])

    ev_tabs = st.tabs([
        f"🟢 Supporting ({len(sup_items)})",
        f"🔴 Contradicting ({len(con_items)})",
        f"🟡 Contextual Differences ({len(ctx_items)})",
    ])

    with ev_tabs[0]:
        if not sup_items:
            st.info("No directly supporting evidence items met the NLI threshold.")
        for item in sup_items:
            _render_evidence_card(item, "supporting")

    with ev_tabs[1]:
        if not con_items:
            st.info("No contradicting evidence items detected in retrieved chunks.")
        for item in con_items:
            _render_evidence_card(item, "contradicting")

    with ev_tabs[2]:
        if not ctx_items:
            st.info("No contextual discrepancies (population, dosage, duration) flagged.")
        for item in ctx_items:
            _render_evidence_card(item, "contextual")


def _render_evidence_card(item: dict, card_type: str):
    """Render a clean, formatted card for an individual evidence passage."""
    title = item.get("title", "Clinical Study")
    year = item.get("year", "?")
    journal = item.get("journal", "PubMed")
    section = item.get("section", "Results")
    study_type = item.get("study_type", "Clinical Trial")
    text = item.get("text", "")
    claim = item.get("claim", {})
    nli = item.get("nli", {})
    ctx = item.get("context_analysis")

    nli_label = nli.get("label", "NEUTRAL")
    nli_conf = nli.get("confidence", 0.0)

    header = f"**{title[:90]}** ({year}, {journal})"
    with st.expander(header, expanded=False):
        cols = st.columns(3)
        cols[0].markdown(f"**Section:** {section}")
        cols[1].markdown(f"**Study Type:** {study_type}")
        cols[2].markdown(f"**DeBERTa NLI:** `{nli_label}` ({nli_conf:.1%})")

        st.markdown("**Evidence Passage:**")
        st.markdown(f"> *{text}*")

        if ctx and ctx.get("differences"):
            st.markdown("**Context Nuances:**")
            for diff in ctx["differences"][:2]:
                st.markdown(f"- **[{diff['dimension'].upper()}]** {diff['description']}")


# ---------------------------------------------------------------------------
# Tab 2: Reasoning Trail (Debug & Viva)
# ---------------------------------------------------------------------------

def render_reasoning_trail(result):
    """Render the full transparent NLP reasoning trail."""
    st.markdown('<div class="section-heading">Detailed Pipeline Execution & NLP Audit Trail</div>', unsafe_allow_html=True)
    st.caption("Inspect each stage of the DiaVeritas pipeline: PICO decomposition, hybrid retrieval, cross-encoder reranking, and DeBERTa-v3 NLI inference.")

    # 1. PICO Query
    with st.expander("Step 1 · Query Normalization & PICO Entity Extraction", expanded=True):
        st.markdown(f"**Original Clinical Question:** `{result.question}`")
        nq = result.normalized_query
        if nq:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Interventions:**\n- " + "\n- ".join(nq.get("interventions", []) or ["None detected"]))
            c2.markdown(f"**Outcomes:**\n- " + "\n- ".join(nq.get("outcomes", []) or ["None detected"]))
            c3.markdown(f"**Population:**\n- " + "\n- ".join(nq.get("population", []) or ["Type 2 Diabetes"]))
            st.markdown(f"**PICO Search String:** `{nq.get('normalized_text', '—')}`")

    # 2. Hybrid Retrieval Candidates
    with st.expander(f"Step 2 · Hybrid Dense + BM25 Retrieval ({len(result.candidates)} candidates)", expanded=False):
        if result.candidates:
            st.markdown("Top 10 Reciprocal Rank Fusion (RRF) candidate chunks before cross-encoder reranking:")
            for item in result.candidates[:10]:
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                c1.markdown(f"**Rank #{item.get('rank', '?')}**")
                c2.metric("Dense Sim", f"{item.get('dense_score', 0):.3f}")
                c3.metric("BM25 Score", f"{item.get('bm25_score', 0):.2f}")
                c4.metric("RRF Score", f"{item.get('rrf_score', 0):.4f}")
                st.markdown(f"> *{item.get('text', '')[:220]}...*")
                st.divider()

    # 3. Cross-Encoder Reranking
    with st.expander(f"Step 3 · Cross-Encoder Reranked Evidence (Top {len(result.reranked)})", expanded=False):
        for item in result.reranked:
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.markdown(f"**Reranker Rank #{item.get('reranker_rank', '?')}**")
            c2.metric("Relevance Score", f"{item.get('reranker_score', 0):.3f}")
            chunk_obj = item.get('chunk')
            sec = getattr(chunk_obj, 'section', 'Results') if chunk_obj else 'Results'
            c3.markdown(f"**Section:** {sec}")
            st.markdown(f"> {item.get('text', '')[:280]}...")
            st.divider()

    # 4. DeBERTa NLI Classification
    with st.expander("Step 4 · Natural Language Inference (MoritzLaurer/DeBERTa-v3)", expanded=False):
        nli_sum = result.nli_summary
        counts = nli_sum.get("counts", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entailments", counts.get("ENTAILMENT", 0))
        c2.metric("Contradictions", counts.get("CONTRADICTION", 0))
        c3.metric("Neutral", counts.get("NEUTRAL", 0))
        c4.metric("Avg Confidence", f"{nli_sum.get('avg_confidence', 0):.1%}")

    # 5. Status Computation Formula
    with st.expander("Step 5 · Evidence Synthesis Calibration Formula", expanded=False):
        s = result.status_result_dict
        st.markdown(f"""
        - **Entailment Ratio:** `{s.get('entailment_ratio', 0):.1%}` (Entailments / Non-neutral claims)
        - **Contradiction Ratio:** `{s.get('contradiction_ratio', 0):.1%}` (Contradictions / Non-neutral claims)
        - **Calculated Status:** `{s.get('status', 'INCONCLUSIVE')}`
        - **Confidence Score:** `{s.get('confidence_score', 0):.2f}` ({s.get('confidence_label', 'Low')})
        """)


# ---------------------------------------------------------------------------
# Tab 3: Literature Corpus Explorer
# ---------------------------------------------------------------------------

def render_corpus_explorer(result):
    """Render literature bibliography and cited studies."""
    st.markdown('<div class="section-heading">Indexed Literature & Cited Studies</div>', unsafe_allow_html=True)
    st.caption(f"All citations grounded in the local {get_chunk_count():,} chunk PubMed/PMC clinical corpus.")

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
        st.info("No papers cited in this query.")
        return

    for pid, item in seen_papers.items():
        authors = item.get("authors", ["Not reported"])
        year = item.get("year", "?")
        title = item.get("title", "Clinical Study")
        journal = item.get("journal", "PubMed")
        study_type = item.get("study_type", "Clinical Trial")
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."

        st.markdown(f"""
        <div class="source-card">
            <div class="source-title">{title}</div>
            <div class="source-meta"><b>{author_str}</b> ({year}) · <i>{journal}</i></div>
            <div class="source-badges">
                <span class="source-badge">PMID: {pid}</span>
                <span class="source-badge">Design: {study_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------------------------

def main():
    render_header()

    # Pre-warm models asynchronously in background so page UI renders immediately
    if "pipeline_warming_started" not in st.session_state:
        st.session_state["pipeline_warming_started"] = True
        threading.Thread(target=load_pipeline, daemon=True).start()

    # Initialize query session state if not set
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = "Is metformin more effective than lifestyle changes for prediabetes?"

    # Quick sample queries
    st.markdown('<div class="quick-query-header">Sample Clinical Questions (Click to autofill):</div>', unsafe_allow_html=True)
    q_cols = st.columns(4)
    for i, sample in enumerate(SAMPLE_QUERIES):
        with q_cols[i]:
            if st.button(sample["label"], key=f"btn_sample_{i}", use_container_width=True):
                st.session_state["user_query"] = sample["query"]
                st.rerun()

    # Main Query Input Box
    query_text = st.text_area(
        "Clinical Research Question",
        value=st.session_state.get("user_query", ""),
        height=85,
        placeholder="Enter a clinical question regarding T2D interventions, outcomes, or drug combinations...",
        key="main_query_input",
    )

    # Action Buttons
    col_run, col_baseline, col_clear, _ = st.columns([1.5, 1.2, 0.8, 3])
    run_full = col_run.button("🔍 Analyze Evidence", type="primary", use_container_width=True)
    run_baseline = col_baseline.button("⚡ Baseline RAG", use_container_width=True)
    clear_btn = col_clear.button("↺ Reset", use_container_width=True)

    if clear_btn:
        st.session_state.pop("last_result", None)
        st.session_state["user_query"] = ""
        st.rerun()

    # Execution logic
    if (run_full or run_baseline) and query_text.strip():
        mode = "baseline" if run_baseline else "diaveritias"

        with st.spinner("Loading evidence pipeline & models..."):
            synthesizer, pipeline_error = load_pipeline()

        if pipeline_error:
            st.error(f"**Pipeline initialization error:** {pipeline_error}")
            return

        progress_bar = st.empty()
        with progress_bar.container():
            st.markdown("**Running DiaVeritas Evidence Synthesis Pipeline...**")
            for step_i in range(5):
                render_pipeline_steps(step_i)
                time.sleep(0.08)

        with st.spinner("Retrieving, verifying NLI, and generating clinical synthesis..."):
            result = synthesizer.run(query_text.strip(), mode=mode)

        progress_bar.empty()
        st.session_state["last_result"] = result

    # Results Display
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        st.markdown("<hr style='margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
        
        # Summary verdict banner
        render_verdict_banner(result)

        # Tabbed Output Interface
        tab_synthesis, tab_trail, tab_corpus = st.tabs([
            "📋 Clinical Synthesis & Evidence",
            "🔬 Pipeline Reasoning Trail",
            "📚 Literature Bibliography",
        ])

        with tab_synthesis:
            render_clinical_synthesis(result)

        with tab_trail:
            render_reasoning_trail(result)

        with tab_corpus:
            render_corpus_explorer(result)

        # Footer latency info
        st.markdown(f"""
        <div class="dv-footer">
            Pipeline: <b>{result.pipeline_mode}</b> · Execution Latency: <b>{result.latency_seconds}s</b> · 
            Embeddings: <code>{settings.embedding_model}</code> · NLI: <code>{settings.nli_model}</code> · 
            LLM: <code>{settings.llm_model}</code>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🩺</div>
            <div style="font-size: 1.1rem; font-weight: 500; color: #1e293b; margin-bottom: 0.3rem;">
                Ready to Analyze Clinical Evidence
            </div>
            <div style="font-size: 0.9rem; color: #64748b; max-width: 600px; margin: 0 auto;">
                Select one of the sample inquiries above or enter a question regarding Type 2 Diabetes interventions (e.g. Metformin, SGLT2 inhibitors, GLP-1 agonists) to retrieve and evaluate peer-reviewed evidence.
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
