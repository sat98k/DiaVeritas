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

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

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
    weighted_support: float = 0.0      # GRADE-weighted support score
    weighted_contradiction: float = 0.0 # GRADE-weighted contradiction score
    confidence_breakdown: Dict[str, float] = field(default_factory=dict) # Inspectable factor breakdown (FR-16.7)

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
            "weighted_support": round(self.weighted_support, 2),
            "weighted_contradiction": round(self.weighted_contradiction, 2),
            "confidence_breakdown": {k: round(v, 3) for k, v in self.confidence_breakdown.items()},
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
    Determine the overall evidence status from the relationship summary using
    GRADE methodological weighting (FR-15.4) and multi-factor confidence scoring (FR-16.6).

    Args:
        summary: EvidenceRelationshipSummary from relationship.py.
        avg_nli_confidence: Average NLI confidence across all pairs.
        avg_retrieval_relevance: Average reranker score.

    Returns:
        EvidenceStatusResult with status, confidence, GRADE weights, and breakdown.
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

    # GRADE-weighted aggregation (FR-15.4)
    def _item_weight(item) -> float:
        if item and hasattr(item, "grade") and item.grade and hasattr(item.grade, "weight"):
            return float(item.grade.weight)
        return 1.0

    w_sup = sum(_item_weight(it) for it in summary.supporting)
    w_con = sum(_item_weight(it) for it in summary.contradicting)
    w_ctx = sum(_item_weight(it) * 0.5 for it in summary.contextual)
    w_effective_con = w_con + w_ctx
    w_decisive = w_sup + w_effective_con

    # Raw ratios
    effective_contradiction = n_con + (n_ctx * 0.5)
    n_decisive = n_sup + effective_contradiction

    total_entailment_ratio = n_sup / n_total if n_total > 0 else 0.0
    total_contradiction_ratio = effective_contradiction / n_total if n_total > 0 else 0.0
    contextual_ratio = n_ctx / n_total if n_total > 0 else 0.0

    decisive_support_ratio = (n_sup / n_decisive) if n_decisive > 0 else 0.0
    decisive_contradict_ratio = (effective_contradiction / n_decisive) if n_decisive > 0 else 0.0

    w_support_ratio = (w_sup / w_decisive) if w_decisive > 0 else 0.0
    w_contradict_ratio = (w_effective_con / w_decisive) if w_decisive > 0 else 0.0

    entailment_ratio = decisive_support_ratio if n_decisive > 0 else total_entailment_ratio
    contradiction_ratio = decisive_contradict_ratio if n_decisive > 0 else total_contradiction_ratio

    # --- Status determination ---
    min_support = settings.supported_min_entailment_ratio
    min_refute = settings.refuted_min_contradiction_ratio

    has_grade_assessments = any(
        it and hasattr(it, "grade") and it.grade is not None
        for it in (summary.supporting + summary.contradicting + summary.contextual)
    )

    # 1. GRADE-weighted decisive check (FR-15.4 & FR-16.4: High certainty consensus with minimal opposing evidence)
    is_supported_grade = (
        has_grade_assessments
        and w_sup >= 3.0
        and w_support_ratio >= 0.60
        and (w_sup >= 1.5 * w_effective_con or n_con == 0)
    )
    is_refuted_grade = (
        has_grade_assessments
        and w_effective_con >= 3.0
        and w_contradict_ratio >= 0.60
        and (w_effective_con >= 1.5 * w_sup or n_sup == 0)
    )

    # 2. Count-based fallback check (when GRADE weights are tied or unassigned)
    is_supported_count = (
        (total_entailment_ratio >= min_support and total_contradiction_ratio < 0.2)
        or (n_decisive >= 2 and n_sup >= 2 and decisive_support_ratio >= 0.7 and effective_contradiction < max(1.0, n_sup * 0.3))
    )
    is_refuted_count = (
        (total_contradiction_ratio >= min_refute and total_entailment_ratio < 0.2)
        or (n_decisive >= 2 and effective_contradiction >= 2 and decisive_contradict_ratio >= 0.7 and n_sup < max(1.0, effective_contradiction * 0.3))
    )

    if is_supported_grade or (is_supported_count and not is_refuted_grade):
        status = SUPPORTED
        rationale = (
            f"Evidence supports this claim with GRADE-weighted strength {w_sup:.1f} vs {w_effective_con:.1f} "
            f"({w_support_ratio:.1%} weighted agreement across {n_sup} supporting items). "
            f"Contradicting evidence is minimal ({n_con} contradictions, {n_ctx} contextual differences)."
        )
    elif is_refuted_grade or (is_refuted_count and not is_supported_grade):
        status = REFUTED
        rationale = (
            f"Evidence contradicts this claim with GRADE-weighted strength {w_effective_con:.1f} vs {w_sup:.1f} "
            f"({w_contradict_ratio:.1%} weighted disagreement across {n_con} contradicting items). "
            f"Supporting evidence is minimal ({n_sup} supporting items)."
        )
    else:
        status = INCONCLUSIVE
        if n_decisive == 0:
            rationale = (
                f"All {n_neu} retrieved evidence pieces provided neutral background context without decisive "
                f"support or contradiction. The evidence cannot be confidently classified as SUPPORTED or REFUTED."
            )
        elif n_sup < 2 and effective_contradiction < 2 and w_decisive < 3.0:
            rationale = (
                f"Evidence is insufficient: only {n_sup} supporting and {n_con} contradicting pieces "
                f"(GRADE strength: {w_decisive:.1f}). Minimum robust trial consensus required for verdict."
            )
        else:
            rationale = (
                f"Evidence is mixed or conflicting: {n_sup} supporting (weight: {w_sup:.1f}) vs "
                f"{n_con} contradicting (weight: {w_con:.1f}), with {n_ctx} contextual differences. "
                f"INCONCLUSIVE reflects genuine clinical controversy or population divergence."
            )

    # --- Multi-factor confidence indicator (FR-16.6 & FR-16.7) ---
    confidence_score, breakdown = _compute_confidence(
        status=status,
        entailment_ratio=w_support_ratio if w_decisive > 0 else entailment_ratio,
        contradiction_ratio=w_contradict_ratio if w_decisive > 0 else contradiction_ratio,
        n_total=n_total,
        summary=summary,
        avg_nli_confidence=avg_nli_confidence,
        avg_retrieval_relevance=avg_retrieval_relevance,
    )
    confidence_label = _confidence_label(confidence_score)

    logger.info(
        f"Evidence status: {status} | Confidence: {confidence_label} ({confidence_score:.2f}) | "
        f"GRADE weighted support: {w_sup:.1f}, contradiction: {w_effective_con:.1f}"
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
        weighted_support=round(w_sup, 2),
        weighted_contradiction=round(w_effective_con, 2),
        confidence_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Multi-factor confidence indicator (FR-16.6)
# ---------------------------------------------------------------------------

def _compute_confidence(
    status: str,
    entailment_ratio: float,
    contradiction_ratio: float,
    n_total: int,
    summary: EvidenceRelationshipSummary,
    avg_nli_confidence: float = 0.0,
    avg_retrieval_relevance: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute inspectable multi-factor confidence score deriving from the 7 factors in FR-16.6:
      1. evidence_agreement
      2. nli_certainty
      3. evidence_quality_tier (GRADE)
      4. study_count (distinct papers)
      5. context_match
      6. citation_support
      7. contradiction_penalty
    """
    # 1. Evidence agreement
    if status == SUPPORTED:
        agreement = entailment_ratio
    elif status == REFUTED:
        agreement = contradiction_ratio
    else:
        agreement = max(entailment_ratio, contradiction_ratio) * 0.4

    # 2. NLI certainty
    nli_factor = avg_nli_confidence if avg_nli_confidence > 0 else 0.6

    # 3. Evidence quality tier (average GRADE weight normalized to [0, 1])
    all_items = summary.supporting + summary.contradicting + summary.contextual + summary.neutral
    if all_items:
        weights = [
            float(it.grade.weight) if hasattr(it, "grade") and it.grade else 1.0
            for it in all_items
        ]
        avg_grade_norm = min(1.0, (sum(weights) / len(weights)) / 3.5)
    else:
        avg_grade_norm = 0.5

    # 4. Number of independent studies (distinct paper_ids)
    distinct_papers = len({it.chunk.paper_id for it in all_items if hasattr(it, "chunk") and it.chunk})
    study_count_factor = min(1.0, 0.4 + (distinct_papers / 8.0))

    # 5. Context match: high contextual conflict lowers overall claim confidence
    context_factor = 1.0 - (summary.n_contextual / max(1, n_total) * 0.5)

    # 6. Citation support (chunks with DOI or valid journal metadata)
    valid_citations = sum(
        1 for it in all_items
        if hasattr(it, "chunk") and it.chunk and it.chunk.journal != "Not reported"
    )
    citation_support = valid_citations / max(1, len(all_items))

    # 7. Contradiction penalty
    contradiction_penalty = 0.15 if summary.n_contradicting > 0 and status == SUPPORTED else 0.0

    # Weighted combination
    score = (
        0.30 * agreement +
        0.20 * avg_grade_norm +
        0.15 * nli_factor +
        0.15 * study_count_factor +
        0.10 * context_factor +
        0.10 * citation_support -
        contradiction_penalty
    )
    final_score = min(1.0, max(0.05, score))

    breakdown = {
        "evidence_agreement": round(agreement, 3),
        "evidence_quality_tier": round(avg_grade_norm, 3),
        "nli_certainty": round(nli_factor, 3),
        "independent_study_count": round(study_count_factor, 3),
        "context_consistency": round(context_factor, 3),
        "citation_support": round(citation_support, 3),
        "contradiction_penalty": round(contradiction_penalty, 3),
    }

    return final_score, breakdown


def _confidence_label(score: float) -> str:
    if score >= 0.65:
        return "High"
    elif score >= 0.40:
        return "Moderate"
    else:
        return "Low"
