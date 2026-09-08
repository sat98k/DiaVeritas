# =============================================================================
# src/evaluation/metrics.py
#
# Evaluation metrics for DiaVeritas.
#
# Evaluates what the system claims to solve:
#
# Component-level:
#   - Retrieval relevance (if relevance labels are available)
#   - NLI consistency (agreement between NLI outputs on similar pairs)
#   - Evidence status consistency (same question → same status across runs)
#
# System-level:
#   - Citation presence (does the answer cite sources?)
#   - End-to-end latency
#   - Evidence coverage (how many claim types are retrieved)
#
# We do NOT claim expert-annotated ground truth.
# We use proxy metrics, consistency checks, and manual spot evaluation.
# =============================================================================

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Citation presence
# ---------------------------------------------------------------------------

def citation_presence_score(answer: str) -> float:
    """
    Check if the answer contains citations.

    Looks for patterns like:
    - [Smith 2021, Diabetes Care]
    - (Smith et al., 2021)
    - [1], [2,3]

    Returns:
        Float in [0, 1]: fraction of detected citation-like patterns
        relative to expected minimum (3 citations per paragraph).
    """
    patterns = [
        r"\[[A-Za-z]+ \d{4}",      # [Smith 2021
        r"\([A-Za-z]+ et al\.",     # (Smith et al.
        r"\[\d+\]",                 # [1]
        r"\[Passage \d+\]",         # [Passage 1]
    ]
    matches = 0
    for pat in patterns:
        matches += len(re.findall(pat, answer))

    # Scoring: 0 citations = 0.0, 3+ citations = 1.0
    return min(1.0, matches / 3)


# ---------------------------------------------------------------------------
# Evidence coverage
# ---------------------------------------------------------------------------

def evidence_coverage_score(
    retrieved_chunks: List[Dict],
    query_interventions: List[str],
    query_outcomes: List[str],
) -> Dict[str, float]:
    """
    Estimate evidence coverage: what fraction of query concepts appear
    in the retrieved evidence pool.

    Higher coverage means the retrieval pipeline found relevant evidence
    for the key query concepts.

    Args:
        retrieved_chunks: List of chunk dicts (must have "text" key).
        query_interventions: Interventions identified in the query.
        query_outcomes: Outcomes identified in the query.

    Returns:
        Dict with coverage scores:
        {
            "intervention_coverage": float,   # % of query interventions mentioned
            "outcome_coverage": float,        # % of query outcomes mentioned
            "overall_coverage": float,        # average of the two
        }
    """
    if not retrieved_chunks:
        return {
            "intervention_coverage": 0.0,
            "outcome_coverage": 0.0,
            "overall_coverage": 0.0,
        }

    all_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()

    def _coverage(terms: List[str]) -> float:
        if not terms:
            return 1.0  # No terms to check — vacuously covered
        matched = sum(1 for t in terms if t.lower() in all_text)
        return matched / len(terms)

    i_cov = _coverage(query_interventions)
    o_cov = _coverage(query_outcomes)
    overall = (i_cov + o_cov) / 2

    return {
        "intervention_coverage": round(i_cov, 3),
        "outcome_coverage": round(o_cov, 3),
        "overall_coverage": round(overall, 3),
    }


# ---------------------------------------------------------------------------
# NLI consistency
# ---------------------------------------------------------------------------

def nli_consistency_score(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
) -> float:
    """
    Measure consistency between two DiaVeritas pipeline runs on the same question.

    Compares:
    - Final evidence status (must match: 1.0 if match, 0.0 if not)
    - NLI label distribution (cosine similarity of count vectors)

    Args:
        result_a, result_b: DiaVeritasResult.to_dict() outputs.

    Returns:
        Float in [0, 1]: 1.0 = perfectly consistent, 0.0 = completely inconsistent.
    """
    status_match = 1.0 if result_a.get("status") == result_b.get("status") else 0.0

    # Compare NLI label distributions
    nli_a = result_a.get("nli_summary", {}).get("counts", {})
    nli_b = result_b.get("nli_summary", {}).get("counts", {})

    labels = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
    vec_a = [nli_a.get(l, 0) for l in labels]
    vec_b = [nli_b.get(l, 0) for l in labels]

    total_a = sum(vec_a) or 1
    total_b = sum(vec_b) or 1
    ratio_a = [x / total_a for x in vec_a]
    ratio_b = [x / total_b for x in vec_b]

    # L1 distance between normalized distributions → similarity
    l1_dist = sum(abs(a - b) for a, b in zip(ratio_a, ratio_b))
    nli_dist_sim = 1.0 - (l1_dist / 2)  # normalize to [0, 1]

    consistency = 0.6 * status_match + 0.4 * nli_dist_sim
    return round(consistency, 3)


# ---------------------------------------------------------------------------
# Answer quality proxies
# ---------------------------------------------------------------------------

def answer_has_uncertainty(answer: str) -> bool:
    """
    Check if the answer acknowledges uncertainty.

    Biomedical answers should hedge claims when evidence is conflicting.
    """
    uncertainty_phrases = [
        "may", "might", "suggests", "indicates", "evidence suggests",
        "uncertain", "inconclusive", "conflicting", "inconsistent",
        "some studies", "limited evidence", "further research",
        "not clear", "not conclusive",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in uncertainty_phrases)


def answer_mentions_contradiction(answer: str) -> bool:
    """Check if the answer explicitly mentions contradictory findings."""
    contradiction_phrases = [
        "however", "contradict", "conflict", "on the other hand",
        "in contrast", "inconsistent", "opposite", "disagree",
        "did not", "no significant", "not significant",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in contradiction_phrases)


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def evaluate_result(
    result: Dict[str, Any],
    query_normalized: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run all proxy evaluations on a DiaVeritasResult dict.

    Args:
        result: DiaVeritasResult.to_dict() output.
        query_normalized: normalize_query() output (for coverage metrics).

    Returns:
        Dict with all evaluation scores.
    """
    answer = result.get("answer", "")
    reranked = result.get("n_reranked", 0)

    ev_summary = result.get("evidence_summary", {})
    retrieved_texts = [
        item.get("text", "")
        for item in (
            ev_summary.get("supporting", []) +
            ev_summary.get("contradicting", []) +
            ev_summary.get("contextual", []) +
            ev_summary.get("neutral", [])
        )
    ]
    if not retrieved_texts:
        # Fallback for baseline mode or when evidence_summary is not populated
        items = result.get("reranked", []) or result.get("candidates", [])
        retrieved_texts = [
            item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
            for item in items
        ]

    interventions = []
    outcomes = []
    if query_normalized:
        interventions = query_normalized.get("interventions", [])
        outcomes = query_normalized.get("outcomes", [])

    coverage = evidence_coverage_score(
        [{"text": t} for t in retrieved_texts],
        interventions,
        outcomes,
    )

    return {
        "citation_presence": round(citation_presence_score(answer), 3),
        "has_uncertainty_language": answer_has_uncertainty(answer),
        "mentions_contradiction": answer_mentions_contradiction(answer),
        "n_evidence_retrieved": reranked,
        "evidence_coverage": coverage,
        "evidence_status": result.get("status", "INCONCLUSIVE"),
        "latency_seconds": result.get("latency_seconds", 0.0),
        "pipeline_mode": result.get("pipeline_mode", "unknown"),
    }
