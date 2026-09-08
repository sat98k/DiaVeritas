# =============================================================================
# src/evidence/status.py
#
# Evidence status determination: SUPPORTED | REFUTED | INCONCLUSIVE
#
# The evidence status is the overall verdict on whether the retrieved
# literature supports, refutes, or cannot resolve the query claim.
#
# IMPORTANT:
#   These labels are evidence status, NOT clinical truth.
#   They reflect what the retrieved corpus says relative to the query.
#   They are limited by corpus coverage, retrieval quality, and NLI accuracy.
#
#   INCONCLUSIVE is a valid and important outcome — it means the evidence
#   is genuinely conflicting, insufficient, or cannot be confidently
#   interpreted. It must NOT be treated as a system failure.
#
# Definitions:
#   SUPPORTED:    A clear majority of evidence supports the claim,
#                 with minimal strong contradiction.
#   REFUTED:      A clear majority of evidence contradicts the claim,
#                 with minimal strong support.
#   INCONCLUSIVE: Evidence is conflicting, insufficient, or too mixed
#                 to support either SUPPORTED or REFUTED.
#
# The thresholds for SUPPORTED/REFUTED are configurable in config.py
# and should be treated as experimental parameters.
#
# Also provides a basic prototype system confidence indicator.
# This confidence is NOT clinically calibrated.
# =============================================================================

from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass

from loguru import logger

from src.config import settings
from src.evidence.relationship import EvidenceRelationshipSummary


# ---------------------------------------------------------------------------
# Evidence status labels
# ---------------------------------------------------------------------------

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class EvidenceStatusResult:
    """
    The final evidence status verdict with supporting metrics.
    """
    status: str                        # SUPPORTED | REFUTED | INCONCLUSIVE
    confidence_score: float            # Prototype system confidence [0, 1]
    confidence_label: str              # "Low" | "Moderate" | "High"
    entailment_ratio: float
    contradiction_ratio: float
    contextual_ratio: float
    n_total: int
    n_supporting: int
    n_contradicting: int
    n_contextual: int
    n_neutral: int
    rationale: str                     # Human-readable explanation of the verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "entailment_ratio": self.entailment_ratio,
            "contradiction_ratio": self.contradiction_ratio,
            "contextual_ratio": self.contextual_ratio,
            "n_total": self.n_total,
            "n_supporting": self.n_supporting,
            "n_contradicting": self.n_contradicting,
            "n_contextual": self.n_contextual,
            "n_neutral": self.n_neutral,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Status determination
# ---------------------------------------------------------------------------

def determine_evidence_status(
    summary: EvidenceRelationshipSummary,
    avg_nli_confidence: float = 0.0,
    avg_retrieval_relevance: float = 0.0,
) -> EvidenceStatusResult:
    """
    Determine the overall evidence status from the relationship summary.

    Args:
        summary: EvidenceRelationshipSummary from relationship.py.
        avg_nli_confidence: Average NLI confidence across all pairs.
                            Used for the confidence indicator.
        avg_retrieval_relevance: Average reranker score.
                                  Used for the confidence indicator.

    Returns:
        EvidenceStatusResult with status, confidence, and rationale.
    """
    n_total = summary.total
    n_sup = summary.n_supporting
    n_con = summary.n_contradicting
    n_ctx = summary.n_contextual
    n_neu = summary.n_neutral

    if n_total == 0:
        return EvidenceStatusResult(
            status=INCONCLUSIVE,
            confidence_score=0.0,
            confidence_label="Low",
            entailment_ratio=0.0,
            contradiction_ratio=0.0,
            contextual_ratio=0.0,
            n_total=0,
            n_supporting=0,
            n_contradicting=0,
            n_contextual=0,
            n_neutral=0,
            rationale="No evidence retrieved. Status is INCONCLUSIVE by default.",
        )

    # Compute ratios
    # "Contextual differences" are counted as partial contradictions
    # because they represent conflicting claims even if context differs
    effective_contradiction = n_con + (n_ctx * 0.5)
    entailment_ratio = n_sup / n_total
    contradiction_ratio = effective_contradiction / n_total
    contextual_ratio = n_ctx / n_total

    # --- Status determination ---
    min_support = settings.supported_min_entailment_ratio
    min_refute = settings.refuted_min_contradiction_ratio

    if entailment_ratio >= min_support and contradiction_ratio < 0.2:
        status = SUPPORTED
        rationale = (
            f"{n_sup}/{n_total} evidence pieces support this claim "
            f"(entailment ratio: {entailment_ratio:.1%}). "
            f"Contradicting evidence is minimal ({n_con} contradictions, "
            f"{n_ctx} contextual differences)."
        )
    elif contradiction_ratio >= min_refute and entailment_ratio < 0.2:
        status = REFUTED
        rationale = (
            f"{n_con}/{n_total} evidence pieces contradict this claim "
            f"(contradiction ratio: {contradiction_ratio:.1%}). "
            f"Supporting evidence is minimal ({n_sup} supporting)."
        )
    else:
        status = INCONCLUSIVE
        parts = []
        if n_sup > 0:
            parts.append(f"{n_sup} supporting")
        if n_con > 0:
            parts.append(f"{n_con} contradicting")
        if n_ctx > 0:
            parts.append(f"{n_ctx} with contextual differences")
        if n_neu > 0:
            parts.append(f"{n_neu} neutral")
        rationale = (
            f"Evidence is mixed or insufficient: {', '.join(parts)}. "
            f"The evidence cannot be confidently classified as SUPPORTED or REFUTED. "
            f"INCONCLUSIVE is a valid finding — it reflects genuine uncertainty "
            f"in the available evidence, not a system failure."
        )

    # --- Prototype confidence indicator ---
    confidence_score = _compute_confidence(
        status=status,
        entailment_ratio=entailment_ratio,
        contradiction_ratio=contradiction_ratio,
        n_total=n_total,
        avg_nli_confidence=avg_nli_confidence,
        avg_retrieval_relevance=avg_retrieval_relevance,
    )
    confidence_label = _confidence_label(confidence_score)

    logger.info(
        f"Evidence status: {status} | Confidence: {confidence_label} ({confidence_score:.2f}) | "
        f"Supporting: {n_sup}, Contradicting: {n_con}, Contextual: {n_ctx}, Neutral: {n_neu}"
    )

    return EvidenceStatusResult(
        status=status,
        confidence_score=round(confidence_score, 3),
        confidence_label=confidence_label,
        entailment_ratio=round(entailment_ratio, 3),
        contradiction_ratio=round(contradiction_ratio, 3),
        contextual_ratio=round(contextual_ratio, 3),
        n_total=n_total,
        n_supporting=n_sup,
        n_contradicting=n_con,
        n_contextual=n_ctx,
        n_neutral=n_neu,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Prototype confidence indicator
# ---------------------------------------------------------------------------

def _compute_confidence(
    status: str,
    entailment_ratio: float,
    contradiction_ratio: float,
    n_total: int,
    avg_nli_confidence: float,
    avg_retrieval_relevance: float,
) -> float:
    """
    Compute a basic prototype system confidence score.

    Inputs:
    - Agreement: how strong is the dominant relationship ratio
    - NLI confidence: average model confidence across pairs
    - Evidence count: more evidence = slightly higher confidence
    - Retrieval relevance: how relevant were the retrieved chunks

    THIS IS NOT CLINICALLY CALIBRATED.
    It is a research prototype indicator only.
    """
    # Agreement score: how dominant is the winning relationship
    if status == SUPPORTED:
        agreement = entailment_ratio
    elif status == REFUTED:
        agreement = contradiction_ratio
    else:
        # INCONCLUSIVE: lower confidence by design
        agreement = max(entailment_ratio, contradiction_ratio) * 0.5

    # Evidence count factor: scale [1, 20] → [0.5, 1.0]
    count_factor = min(1.0, 0.5 + (n_total / 40))

    # NLI confidence factor
    nli_factor = avg_nli_confidence if avg_nli_confidence > 0 else 0.5

    # Retrieval relevance factor (normalized; cross-encoder scores can be negative)
    # Clamp to [0, 1]
    rel_factor = min(1.0, max(0.0, (avg_retrieval_relevance + 10) / 20))

    # Weighted combination (these weights are experimental)
    score = (
        0.40 * agreement +
        0.25 * nli_factor +
        0.20 * count_factor +
        0.15 * rel_factor
    )

    return min(1.0, max(0.0, score))


def _confidence_label(score: float) -> str:
    if score >= 0.65:
        return "High"
    elif score >= 0.40:
        return "Moderate"
    else:
        return "Low"
