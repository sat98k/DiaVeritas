# =============================================================================
# src/generation/synthesizer.py
#
# Full pipeline orchestrator — the main entry point for the online pipeline.
#
# This module ties together all pipeline stages:
#   Query normalization
#   → Hybrid retrieval (dense + BM25 + RRF)
#   → Cross-encoder reranking
#   → Claim extraction
#   → Biomedical NLI
#   → Contextual contradiction analysis
#   → Evidence relationship derivation
#   → Evidence status determination
#   → LLM synthesis
#   → Citation-grounded answer
#
# Also supports:
#   - Baseline RAG mode (settings.baseline_mode = True)
#   - Ablation: skip NLI/context components
#
# The synthesizer returns a DiaVeritasResult with the full reasoning trail,
# which is used by both the clean UI view and the research/debug view.
# =============================================================================

from __future__ import annotations

import time
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from loguru import logger

from src.config import settings
from src.claims.query_normalizer import normalize_query
from src.claims.claim_extractor import extract_claims, StructuredClaim
from src.claims.claim_grouper import ClaimGrouper, ClaimGroup
from src.nli.nli_classifier import (
    classify_evidence_against_query, summarize_nli_results, NLIResult
)
from src.context.contradiction_analyzer import analyze_contradictions
from src.evidence.relationship import build_evidence_items, summarize_evidence_relationships, EvidenceItem
from src.evidence.grader import grade_grader
from src.evidence.status import determine_evidence_status
from src.generation.prompt_builder import (
    build_synthesis_prompt, build_baseline_prompt, build_descriptive_qa_prompt,
    SYSTEM_PROMPT, DESCRIPTIVE_QA_SYSTEM_PROMPT
)


# ---------------------------------------------------------------------------
# Result structure — the full reasoning trail
# ---------------------------------------------------------------------------

@dataclass
class DiaVeritasResult:
    """
    Complete result from the DiaVeritas pipeline.

    Contains all intermediate artifacts so the debug view can show
    the full reasoning trail from query to answer.
    """
    # Input
    question: str
    normalized_query: Dict[str, Any] = field(default_factory=dict)
    query_type: str = "VERIFICATION"

    # Retrieval
    candidates: List[Dict] = field(default_factory=list)    # RRF candidates
    reranked: List[Dict] = field(default_factory=list)       # After reranking

    # Analysis
    claims: List[StructuredClaim] = field(default_factory=list)
    claim_groups: List[Dict[str, Any]] = field(default_factory=list)
    nli_summary: Dict[str, Any] = field(default_factory=dict)
    evidence_summary_dict: Dict[str, Any] = field(default_factory=dict)
    status: str = "INCONCLUSIVE"
    status_result_dict: Dict[str, Any] = field(default_factory=dict)

    # Answer & Verification
    answer: str = ""
    answer_error: str = ""
    ungrounded_claims: List[str] = field(default_factory=list)
    stripped_claims: List[str] = field(default_factory=list)

    # Metadata
    pipeline_mode: str = "diaveritias"   # "diaveritias" | "baseline"
    latency_seconds: float = 0.0
    models_used: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "normalized_query": self.normalized_query,
            "query_type": self.query_type,
            "n_candidates": len(self.candidates),
            "n_reranked": len(self.reranked),
            "n_claims": len(self.claims),
            "claim_groups": self.claim_groups,
            "nli_summary": self.nli_summary,
            "evidence_summary": self.evidence_summary_dict,
            "status": self.status,
            "status_result": self.status_result_dict,
            "answer": self.answer,
            "answer_error": self.answer_error,
            "ungrounded_claims": self.ungrounded_claims,
            "stripped_claims": self.stripped_claims,
            "pipeline_mode": self.pipeline_mode,
            "latency_seconds": self.latency_seconds,
            "models_used": self.models_used,
        }


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class Synthesizer:
    """
    Orchestrates the full DiaVeritas online pipeline.

    Usage:
        synth = Synthesizer(embedder, vector_store, bm25_index, llm_client)
        result = synth.run("Does metformin reduce cardiovascular risk in T2D?")
    """

    def __init__(
        self,
        embedder,
        vector_store,
        bm25_index,
        llm_client=None,
        reranker=None,
    ):
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import CrossEncoderReranker

        self._hybrid = HybridRetriever(embedder, vector_store, bm25_index)
        self._reranker = reranker or CrossEncoderReranker()
        self._llm = llm_client
        if self._llm is not None:
            try:
                if hasattr(self._llm, "_load"):
                    self._llm._load()
            except Exception as e:
                logger.warning(
                    f"LLM client could not be initialized ({e}). "
                    "Running in structured fallback mode without LLM."
                )
                self._llm = None
        self._models_used = {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "nli": settings.nli_model,
            "llm": f"{settings.llm_provider}/{settings.llm_model}",
        }

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: str = "diaveritias",
    ) -> DiaVeritasResult:
        """
        Run the full pipeline for a user question.

        Args:
            question: The user's natural language question.
            top_k: Override for reranker_top_k (default: settings.reranker_top_k).
            mode: "diaveritias" (full pipeline) or "baseline" (plain RAG).

        Returns:
            DiaVeritasResult with full reasoning trail.
        """
        start_time = time.time()
        result = DiaVeritasResult(
            question=question,
            pipeline_mode=mode,
            models_used=self._models_used,
        )

        if mode == "baseline" or settings.baseline_mode:
            return self._run_baseline(question, result, start_time)

        return self._run_diaveritias(question, result, start_time, top_k)

    # -----------------------------------------------------------------------
    # Full DiaVeritas pipeline
    # -----------------------------------------------------------------------

    def _run_diaveritias(self, question, result, start_time, top_k):
        """Execute the full evidence analysis pipeline."""
        top_k = top_k or settings.reranker_top_k

        try:
            # Step 1: Query normalization & Intent Classification
            logger.info("[Step 1] Normalizing query & identifying intent...")
            result.normalized_query = normalize_query(question)
            result.query_type = result.normalized_query.get("query_type", "VERIFICATION")
            search_query = result.normalized_query["normalized_text"]
            dense_query = result.normalized_query.get("dense_query", question)

            # Step 2: Hybrid retrieval
            logger.info(f"[Step 2] Hybrid retrieval (dense + BM25 + RRF) for intent '{result.query_type}'...")
            candidates = self._hybrid.retrieve(
                query=question,
                dense_query=dense_query,
                bm25_query=search_query,
            )
            result.candidates = candidates

            # Step 3: Reranking
            logger.info(f"[Step 3] Cross-encoder reranking (top-{top_k})...")
            reranked = self._reranker.rerank(question, candidates, top_k=top_k)
            result.reranked = reranked

            if not reranked:
                logger.warning("No evidence retrieved. Returning INCONCLUSIVE.")
                result.answer = (
                    "No relevant evidence was retrieved from the corpus for this question. "
                    "The corpus may not contain relevant papers, or the index may be empty. "
                    "Please run build_index.py to populate the corpus."
                )
                result.latency_seconds = time.time() - start_time
                return result

            # Extract Chunk objects from reranked results
            chunks = [item["chunk"] for item in reranked]

            # Step 4: Claim extraction
            logger.info("[Step 4] Extracting structured claims...")
            claims = extract_claims(chunks, llm_client=None)
            result.claims = claims

            # Step 4b: Claim Grouping & Concept Gate (Module 12: FR-12.1 to FR-12.4)
            logger.info("[Step 4b] Grouping claims and filtering off-target concepts...")
            grouper = ClaimGrouper()
            on_target_claims, neutral_gated_tuples, all_groups = grouper.group_and_filter_claims(
                claims, result.normalized_query
            )
            result.claim_groups = [g.to_dict() for g in all_groups]

            # Step 5: Biomedical NLI (Module 13)
            logger.info(f"[Step 5] Running biomedical NLI on {len(on_target_claims)} on-target claims...")
            query_claim_text = (
                result.normalized_query.get("hypothesis_text")
                or result.normalized_query.get("normalized_text", question)
            )

            # NLI for on-target claims
            on_target_nli = classify_evidence_against_query(query_claim_text, on_target_claims)
            on_target_map = {res.premise_chunk_id: res for res in on_target_nli}

            # Off-target gated claims are deterministically assigned NEUTRAL (FR-12.3)
            neutral_gated_map = {
                c.chunk_id: NLIResult(
                    premise_chunk_id=c.chunk_id,
                    hypothesis_text=query_claim_text,
                    label="NEUTRAL",
                    confidence=0.85,
                    all_scores={"ENTAILMENT": 0.05, "CONTRADICTION": 0.10, "NEUTRAL": 0.85},
                    premise_text=f"Off-target concept: {reason}",
                )
                for c, reason in neutral_gated_tuples
            }

            # Align NLI results strictly with original chunks/claims order
            nli_results: List[NLIResult] = []
            for c in claims:
                res = on_target_map.get(c.chunk_id) or neutral_gated_map.get(c.chunk_id)
                if res is None:
                    res = NLIResult(c.chunk_id, query_claim_text, "NEUTRAL", 0.75, {})
                nli_results.append(res)

            nli_summary = summarize_nli_results(nli_results)
            result.nli_summary = {
                k: v for k, v in nli_summary.items()
                if k not in ("entailments", "contradictions", "neutrals")
            }

            # Step 6: Contextual contradiction analysis (Module 14: FR-14.1 to FR-14.5)
            logger.info("[Step 6] Analyzing contextual contradictions...")
            target_ints = result.normalized_query.get("interventions", [])
            target_outs = result.normalized_query.get("outcomes", [])
            target_pop = result.normalized_query.get("population", [])
            if not target_pop and result.normalized_query.get("disease"):
                target_pop = result.normalized_query.get("disease")
            ref_claim = StructuredClaim(
                chunk_id="query_ref",
                paper_id="query",
                section="Query",
                intervention=target_ints[0] if target_ints else "Target Intervention",
                outcome=target_outs[0] if target_outs else "Target Outcome",
                population=target_pop[0] if target_pop else "Type 2 Diabetes",
                direction="Reduction" if any(w in question.lower() for w in ["reduce", "lower", "decrease", "prevent"]) else "Increase",
                raw_text=question,
            )
            context_analyses = analyze_contradictions(claims, nli_results, reference_claim=ref_claim)

            # Step 7: GRADE / ADA evidence certainty grading & relationships (Module 15: FR-15.1 to FR-15.4)
            logger.info("[Step 7] Grading evidence certainty (GRADE/ADA) and deriving relationships...")
            grades = [grade_grader.grade_evidence(c, cl) for c, cl in zip(chunks, claims)]
            evidence_items = build_evidence_items(
                chunks, claims, nli_results, context_analyses, grades=grades
            )
            ev_summary = summarize_evidence_relationships(evidence_items)
            result.evidence_summary_dict = ev_summary.to_dict()

            # Step 8: Evidence status determination (Module 16: FR-16.1 to FR-16.7)
            logger.info("[Step 8] Determining evidence status with GRADE weighting...")
            avg_nli_conf = nli_summary.get("avg_confidence", 0.0)
            avg_reranker = (
                sum(r.get("reranker_score", 0.0) for r in reranked) / len(reranked)
                if reranked else 0.0
            )
            status_result = determine_evidence_status(
                ev_summary, avg_nli_conf, avg_reranker
            )
            result.status = status_result.status
            result.status_result_dict = status_result.to_dict()

            # Step 9: LLM answer synthesis (Module 17: Dual-Mode Answering)
            logger.info(f"[Step 9] LLM synthesis (Mode: {result.query_type})...")
            if result.query_type == "DESCRIPTIVE_QA":
                prompt = build_descriptive_qa_prompt(
                    question=question,
                    normalized_query=result.normalized_query,
                    evidence_items=evidence_items,
                )
                sys_prompt = DESCRIPTIVE_QA_SYSTEM_PROMPT
            else:
                prompt = build_synthesis_prompt(
                    question=question,
                    normalized_query=result.normalized_query,
                    evidence_summary=ev_summary,
                    status_result=status_result,
                )
                sys_prompt = SYSTEM_PROMPT

            if self._llm:
                try:
                    result.answer = self._llm.complete(
                        prompt=prompt,
                        system_prompt=sys_prompt,
                        max_tokens=1500,
                        temperature=0.2,
                    )
                except Exception as e:
                    logger.error(f"LLM synthesis failed: {e}")
                    result.answer_error = str(e)
                    result.answer = _fallback_answer(result, evidence_items)
            else:
                logger.warning("No LLM client configured. Generating fallback answer.")
                result.answer = _fallback_answer(result, evidence_items)

            # Step 10: Grounding check & strict enforcement (FR-17.2, NFR-5.2)
            clean_answer, stripped = enforce_strict_grounding(result.answer, evidence_items)
            result.answer = clean_answer
            result.ungrounded_claims = []
            result.stripped_claims = stripped

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            result.answer_error = str(e)
            result.answer = f"Pipeline error: {e}"

        result.latency_seconds = round(time.time() - start_time, 2)
        logger.info(f"Pipeline complete in {result.latency_seconds}s | Status: {result.status}")
        return result

    # -----------------------------------------------------------------------
    # Baseline RAG pipeline
    # -----------------------------------------------------------------------

    def _run_baseline(self, question, result, start_time):
        """Execute the standard RAG baseline (no NLI/claim/evidence analysis)."""
        logger.info("[Baseline] Running standard RAG pipeline...")
        result.pipeline_mode = "baseline"

        try:
            # Query normalization
            result.normalized_query = normalize_query(question)
            search_query = result.normalized_query["normalized_text"]
            dense_query = result.normalized_query.get("dense_query", question)

            # Dense + BM25 retrieval
            candidates = self._hybrid.retrieve(
                query=question,
                dense_query=dense_query,
                bm25_query=search_query,
            )
            result.candidates = candidates

            # Rerank
            reranked = self._reranker.rerank(question, candidates)
            result.reranked = reranked

            retrieved_texts = [item["text"] for item in reranked]

            # LLM directly from retrieved passages (baseline)
            if self._llm and retrieved_texts:
                prompt = build_baseline_prompt(question, retrieved_texts)
                try:
                    result.answer = self._llm.complete(prompt, max_tokens=1000, temperature=0.3)
                except Exception as e:
                    result.answer_error = str(e)
                    result.answer = f"[Baseline LLM error: {e}]"
            else:
                result.answer = "\n\n".join(
                    f"[{i+1}] {text[:400]}..." for i, text in enumerate(retrieved_texts[:5])
                )

        except Exception as e:
            logger.exception(f"Baseline pipeline error: {e}")
            result.answer_error = str(e)
            result.answer = f"Baseline pipeline error: {e}"

        result.latency_seconds = round(time.time() - start_time, 2)
        return result


# ---------------------------------------------------------------------------
# Strict Sentence-level Grounding Enforcement (FR-17.2, NFR-5.2)
# ---------------------------------------------------------------------------

def _find_cited_chunks(sentence: str, evidence_items: List[EvidenceItem]) -> List[Chunk]:
    """Find which evidence chunk(s) are referenced by citation markers in the sentence."""
    cited = []
    s_lower = sentence.lower()

    for idx, item in enumerate(evidence_items, 1):
        chunk = item.chunk
        # 1. Match [Source N], [Passage N], or [N]
        if f"source {idx}" in s_lower or f"passage {idx}" in s_lower or f"[{idx}]" in sentence:
            cited.append(chunk)
            continue

        # 2. Match PMID / paper_id
        if chunk.paper_id and chunk.paper_id != "Not reported" and chunk.paper_id.lower() in s_lower:
            cited.append(chunk)
            continue

        # 3. Match Author + Year (e.g. "Smith 2022" or "Smith et al. 2022")
        if chunk.authors and chunk.year and chunk.year != "?":
            first_author = chunk.authors[0].split()[0].replace(",", "").lower()
            if len(first_author) >= 3 and first_author in s_lower and str(chunk.year) in s_lower:
                cited.append(chunk)
                continue

        # 4. Match Title substring if in brackets
        if chunk.title and len(chunk.title) >= 10:
            title_stem = chunk.title[:30].lower()
            if title_stem in s_lower:
                cited.append(chunk)
                continue

    return cited


def _passage_supports_sentence(sentence: str, chunk: Chunk) -> bool:
    """Check if a specific evidence chunk semantically/lexically supports a sentence."""
    skip_words = {
        "pmid", "doi", "trial", "study", "studies", "journal",
        "author", "source", "passage", "table", "figure",
    }
    s_words = {
        w for w in re.findall(r"\b[a-z]{4,}\b", sentence.lower())
        if w not in skip_words
    }
    if not s_words:
        return True

    chunk_words = {
        w for w in re.findall(r"\b[a-z]{4,}\b", chunk.text.lower())
        if w not in skip_words
    }

    overlap = s_words & chunk_words
    return len(overlap) >= 3 or (len(s_words) > 0 and (len(overlap) / len(s_words)) >= 0.35)


def enforce_strict_grounding(answer: str, evidence_items: List[EvidenceItem]) -> Tuple[str, List[str]]:
    """
    Enforce strict sentence-level citation grounding (NFR-5.2, FR-17.2).
    Checks every sentence in the synthesized answer. Verifies that sentences with
    citations are actually supported by the specific cited passage(s). Sentences with
    unsupported citations or lacking evidence support are programmatically stripped.

    Returns:
        Tuple of (clean_grounded_answer, list_of_removed_ungrounded_sentences)
    """
    if not answer or "No relevant evidence" in answer or "Pipeline error" in answer:
        return answer, []

    paragraphs = answer.split("\n\n")
    citation_pat = re.compile(r"\[.+?\d{4}.*?\]|\[Source \d+\]|\([A-Za-z]+ et al\.,?\s*\d{4}\)|\[\d+\]")

    cleaned_paragraphs = []
    ungrounded_sentences = []

    for para in paragraphs:
        # If paragraph is a markdown header, list, table, or disclaimer, preserve it
        if para.strip().startswith(("#", "|", "*This analysis is for research", "*(No external LLM", "**Evidence Status", "**Evidence Analysis", "**Verdict Rationale", "**Supporting Evidence", "**Contradicting Evidence")):
            cleaned_paragraphs.append(para)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        valid_sentences = []

        for s in sentences:
            s_clean = s.strip()
            # Skip short fragments, bullet markers, disclaimers
            if len(s_clean) < 35 or s_clean.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.")):
                valid_sentences.append(s)
                continue
            if any(skip in s_clean.lower() for skip in ["medical advice", "for research purposes", "clinical question:", "summary:"]):
                valid_sentences.append(s)
                continue

            # Check citation presence and verify cited passage support
            if citation_pat.search(s_clean):
                cited_chunks = _find_cited_chunks(s_clean, evidence_items)
                if cited_chunks:
                    is_supported_by_cited = any(
                        _passage_supports_sentence(s_clean, c) for c in cited_chunks
                    )
                    if is_supported_by_cited:
                        valid_sentences.append(s)
                        continue
                    else:
                        # Citation is attached, but the cited source does not support this claim
                        ungrounded_sentences.append(s_clean)
                        logger.info(f"Stripped citation-misattributed sentence: '{s_clean[:80]}...'")
                        continue
                else:
                    # Citation pattern found but specific chunk not matched: check general evidence pool
                    has_general_support = any(
                        _passage_supports_sentence(s_clean, item.chunk)
                        for item in evidence_items
                    )
                    if has_general_support:
                        valid_sentences.append(s)
                    else:
                        ungrounded_sentences.append(s_clean)
                        logger.info(f"Stripped ungrounded cited sentence: '{s_clean[:80]}...'")
                    continue

            # Sentence without citation marker: check lexical grounding in evidence chunks
            has_general_overlap = any(
                _passage_supports_sentence(s_clean, item.chunk)
                for item in evidence_items
            )
            if has_general_overlap:
                valid_sentences.append(s)
            else:
                ungrounded_sentences.append(s_clean)
                logger.info(f"Stripped ungrounded clinical sentence: '{s_clean[:80]}...'")

        if valid_sentences:
            cleaned_paragraphs.append(" ".join(valid_sentences))

    filtered_answer = "\n\n".join(cleaned_paragraphs).strip()
    return filtered_answer, ungrounded_sentences


def _verify_grounding(answer: str, evidence_items: List[EvidenceItem]) -> List[str]:
    """Check sentence-level citation grounding in the generated answer (FR-17.2)."""
    _, ungrounded = enforce_strict_grounding(answer, evidence_items)
    return ungrounded


# ---------------------------------------------------------------------------
# Fallback answer (when LLM is unavailable)
# ---------------------------------------------------------------------------

def _fallback_answer(result: DiaVeritasResult, evidence_items: Optional[List[EvidenceItem]] = None) -> str:
    """
    Generate a structured fallback answer from the pipeline results
    when no LLM is available or when LLM synthesis fails.
    Supports both VERIFICATION and DESCRIPTIVE_QA modes.
    """
    items = evidence_items or []

    # 1. Descriptive Q&A fallback
    if result.query_type == "DESCRIPTIVE_QA":
        lines = [
            f"### Clinical Synthesis: {result.question}\n",
            f"*(No external LLM configured — structured pipeline evidence synthesis)*\n",
            f"**Evidence Base Summary:** Retrieved and analyzed {len(items)} clinical literature passages.\n",
            "**Key Findings from Verified Literature:**",
        ]
        for i, item in enumerate(items[:5], 1):
            chunk = item.chunk
            grade_str = item.grade.tier if item.grade else "Standard"
            first_author = chunk.authors[0].split(",")[0] if chunk.authors else "Author"
            citation = f"[{first_author} {chunk.year}, {chunk.journal}]"
            lines.append(f"{i}. **{citation}** ({chunk.study_type}, GRADE Certainty: {grade_str}):")
            lines.append(f"   _{chunk.text[:350]}..._\n")

        lines.append(
            "*This analysis is for research purposes only. "
            "It is not medical advice. To enable natural-language LLM synthesis, "
            "add an API key (e.g., GROQ_API_KEY) to your .env file.*"
        )
        return "\n".join(lines)

    # 2. Verification fallback
    status = result.status
    ev = result.evidence_summary_dict
    stat_dict = result.status_result_dict

    lines = [
        f"**Evidence Status: {status}** (Confidence: {stat_dict.get('confidence_label', 'Moderate')} - {stat_dict.get('confidence_score', 0.0):.1%})",
        f"*(No external LLM configured — showing structured pipeline output)*\n",
        f"**Evidence Analysis:**",
        f"- Supporting: {ev.get('n_supporting', 0)} evidence items (GRADE weighted: {stat_dict.get('weighted_support', 0.0)})",
        f"- Contradicting: {ev.get('n_contradicting', 0)} evidence items (GRADE weighted: {stat_dict.get('weighted_contradiction', 0.0)})",
        f"- Contextual differences: {ev.get('n_contextual', 0)} evidence items",
        f"- Neutral: {ev.get('n_neutral', 0)} evidence items",
        f"\n**Verdict Rationale:** {stat_dict.get('rationale', '')}",
        f"\n**Supporting Evidence:**",
    ]

    for item in ev.get("supporting", [])[:3]:
        title = item.get("title", "Unknown")
        year = item.get("year", "?")
        text = item.get("text", "")[:300]
        grade_info = item.get("grade") or {}
        g_tier = grade_info.get("tier", "Standard")
        lines.append(f"- [{title} ({year}) | GRADE: {g_tier}]: {text}...")

    if ev.get("n_contradicting", 0) > 0:
        lines.append(f"\n**Contradicting Evidence:**")
        for item in ev.get("contradicting", [])[:3]:
            title = item.get("title", "Unknown")
            year = item.get("year", "?")
            text = item.get("text", "")[:300]
            grade_info = item.get("grade") or {}
            g_tier = grade_info.get("tier", "Standard")
            lines.append(f"- [{title} ({year}) | GRADE: {g_tier}]: {text}...")

    if ev.get("n_contextual", 0) > 0:
        lines.append(f"\n**Contextual Differences:**")
        for item in ev.get("contextual", [])[:3]:
            title = item.get("title", "Unknown")
            year = item.get("year", "?")
            ca = item.get("context_analysis") or {}
            diffs = [d.get("description", "") for d in ca.get("differences", [])]
            diff_text = "; ".join(diffs[:2]) if diffs else "Contextual divergence"
            lines.append(f"- [{title} ({year})]: {diff_text}")

    lines.append(
        "\n*This analysis is for research purposes only. "
        "It is not medical advice. To enable LLM synthesis, "
        "add an API key (e.g., GROQ_API_KEY) to your .env file.*"
    )
    return "\n".join(lines)
