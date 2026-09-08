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
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from loguru import logger

from src.config import settings
from src.claims.query_normalizer import normalize_query
from src.claims.claim_extractor import extract_claims, StructuredClaim
from src.nli.nli_classifier import (
    classify_evidence_against_query, summarize_nli_results
)
from src.context.contradiction_analyzer import analyze_contradictions
from src.evidence.relationship import build_evidence_items, summarize_evidence_relationships
from src.evidence.status import determine_evidence_status
from src.generation.prompt_builder import (
    build_synthesis_prompt, build_baseline_prompt, SYSTEM_PROMPT
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

    # Retrieval
    candidates: List[Dict] = field(default_factory=list)    # RRF candidates
    reranked: List[Dict] = field(default_factory=list)       # After reranking

    # Analysis
    claims: List[StructuredClaim] = field(default_factory=list)
    nli_summary: Dict[str, Any] = field(default_factory=dict)
    evidence_summary_dict: Dict[str, Any] = field(default_factory=dict)
    status: str = "INCONCLUSIVE"
    status_result_dict: Dict[str, Any] = field(default_factory=dict)

    # Answer
    answer: str = ""
    answer_error: str = ""

    # Metadata
    pipeline_mode: str = "diaveritias"   # "diaveritias" | "baseline"
    latency_seconds: float = 0.0
    models_used: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "normalized_query": self.normalized_query,
            "n_candidates": len(self.candidates),
            "n_reranked": len(self.reranked),
            "n_claims": len(self.claims),
            "nli_summary": self.nli_summary,
            "evidence_summary": self.evidence_summary_dict,
            "status": self.status,
            "status_result": self.status_result_dict,
            "answer": self.answer,
            "answer_error": self.answer_error,
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
            # Step 1: Query normalization
            logger.info("[Step 1] Normalizing query...")
            result.normalized_query = normalize_query(question)
            search_query = result.normalized_query["normalized_text"]

            # Step 2: Hybrid retrieval
            logger.info("[Step 2] Hybrid retrieval (dense + BM25 + RRF)...")
            candidates = self._hybrid.retrieve(search_query)
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
            claims = extract_claims(chunks, llm_client=self._llm)
            result.claims = claims

            # Step 5: NLI
            logger.info("[Step 5] Running biomedical NLI...")
            query_claim_text = result.normalized_query["normalized_text"]
            nli_results = classify_evidence_against_query(query_claim_text, claims)
            nli_summary = summarize_nli_results(nli_results)
            result.nli_summary = {
                k: v for k, v in nli_summary.items()
                if k not in ("entailments", "contradictions", "neutrals")
            }

            # Step 6: Contextual contradiction analysis
            logger.info("[Step 6] Analyzing contextual contradictions...")
            context_analyses = analyze_contradictions(claims, nli_results)

            # Step 7: Evidence relationships
            logger.info("[Step 7] Deriving evidence relationships...")
            evidence_items = build_evidence_items(
                chunks, claims, nli_results, context_analyses
            )
            ev_summary = summarize_evidence_relationships(evidence_items)
            result.evidence_summary_dict = ev_summary.to_dict()

            # Step 8: Evidence status
            logger.info("[Step 8] Determining evidence status...")
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

            # Step 9: LLM synthesis
            logger.info("[Step 9] LLM synthesis...")
            if self._llm:
                prompt = build_synthesis_prompt(
                    question=question,
                    normalized_query=result.normalized_query,
                    evidence_summary=ev_summary,
                    status_result=status_result,
                )
                try:
                    result.answer = self._llm.complete(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        max_tokens=1500,
                        temperature=0.2,
                    )
                except Exception as e:
                    logger.error(f"LLM synthesis failed: {e}")
                    result.answer_error = str(e)
                    result.answer = _fallback_answer(result)
            else:
                logger.warning("No LLM client configured. Generating fallback answer.")
                result.answer = _fallback_answer(result)

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

            # Dense + BM25 retrieval
            candidates = self._hybrid.retrieve(search_query)
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
# Fallback answer (when LLM is unavailable)
# ---------------------------------------------------------------------------

def _fallback_answer(result: DiaVeritasResult) -> str:
    """
    Generate a structured fallback answer from the pipeline results
    when no LLM is available or when LLM synthesis fails.
    """
    status = result.status
    ev = result.evidence_summary_dict

    lines = [
        f"**Evidence Status: {status}**",
        f"*(No LLM configured — showing structured pipeline output)*\n",
        f"**Evidence Analysis:**",
        f"- Supporting: {ev.get('n_supporting', 0)} evidence items",
        f"- Contradicting: {ev.get('n_contradicting', 0)} evidence items",
        f"- Contextual differences: {ev.get('n_contextual', 0)} evidence items",
        f"- Neutral: {ev.get('n_neutral', 0)} evidence items",
        f"\n**Supporting Evidence:**",
    ]

    for item in ev.get("supporting", [])[:3]:
        title = item.get("title", "Unknown")
        year = item.get("year", "?")
        text = item.get("text", "")[:300]
        lines.append(f"- [{title} ({year})]: {text}...")

    if ev.get("n_contradicting", 0) > 0:
        lines.append(f"\n**Contradicting Evidence:**")
        for item in ev.get("contradicting", [])[:3]:
            title = item.get("title", "Unknown")
            year = item.get("year", "?")
            text = item.get("text", "")[:300]
            lines.append(f"- [{title} ({year})]: {text}...")

    lines.append(
        "\n*This analysis is for research purposes only. "
        "It is not medical advice. To enable LLM synthesis, "
        "add an API key (e.g., GROQ_API_KEY) to your .env file.*"
    )
    return "\n".join(lines)
