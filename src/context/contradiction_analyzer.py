# =============================================================================
# src/context/contradiction_analyzer.py
#
# Contextual contradiction analysis.
#
# When NLI identifies a potential contradiction between two evidence claims,
# this module compares their structured context to identify possible reasons
# why they may differ.
#
# Contextual factors compared:
#   - Population (who the study involved)
#   - Intervention (what exactly was used)
#   - Comparator (what it was compared to)
#   - Outcome (what was measured)
#   - Duration (how long the study ran)
#   - Study design/context (RCT vs. observational, etc.)
#
# CRITICAL DESIGN PRINCIPLE:
#   This module IDENTIFIES potential contextual differences.
#   It does NOT claim to explain WHY studies disagree.
#   It does NOT determine which study is correct.
#   Context analysis interprets NLI-detected conflicts — it does not resolve them.
#
# Output:
#   ContextComparisonResult containing:
#     - list of identified contextual differences
#     - conflict classification: SEMANTIC | CONTEXTUAL | UNRESOLVED
#     - a human-readable summary
# =============================================================================

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

from loguru import logger

from src.claims.claim_extractor import StructuredClaim
from src.nli.nli_classifier import NLIResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContextualDifference:
    """A single identified contextual difference between two claims."""
    dimension: str        # "population" | "intervention" | "outcome" | etc.
    claim_a_value: str
    claim_b_value: str
    description: str      # Human-readable description of the difference


@dataclass
class ContextComparisonResult:
    """
    Result of comparing context between a contradicting pair of claims.
    """
    claim_a_id: str
    claim_b_id: str
    nli_label: str                    # The NLI label that triggered this analysis
    nli_confidence: float

    # Identified contextual differences
    differences: List[ContextualDifference] = field(default_factory=list)

    # Classification of the conflict type
    # SEMANTIC: claims make genuinely opposite statements with similar context
    # CONTEXTUAL: differences in study context may explain the apparent conflict
    # UNRESOLVED: insufficient context to determine
    conflict_type: str = "UNRESOLVED"

    # Human-readable summary
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Context comparison
# ---------------------------------------------------------------------------

def compare_claim_context(
    claim_a: StructuredClaim,
    claim_b: StructuredClaim,
    nli_result: NLIResult,
) -> ContextComparisonResult:
    """
    Compare the context of two claims that have a CONTRADICTION NLI relationship.

    Args:
        claim_a: First claim (typically the query-aligned claim)
        claim_b: Second claim (the contradicting evidence claim)
        nli_result: The NLI result that identified the contradiction

    Returns:
        ContextComparisonResult with identified differences and conflict type.
    """
    differences: List[ContextualDifference] = []

    # Compare each contextual dimension
    dimensions = [
        ("population", claim_a.population, claim_b.population,
         "Studies examined different populations"),
        ("intervention", claim_a.intervention, claim_b.intervention,
         "Studies examined different interventions"),
        ("comparator", claim_a.comparator, claim_b.comparator,
         "Studies used different comparators"),
        ("outcome", claim_a.outcome, claim_b.outcome,
         "Studies measured different outcomes"),
        ("duration", claim_a.duration, claim_b.duration,
         "Studies had different follow-up durations"),
        ("study_context", claim_a.study_context, claim_b.study_context,
         "Studies differed in design or context"),
    ]

    for dim, val_a, val_b, base_desc in dimensions:
        diff = _compare_dimension(dim, val_a, val_b, base_desc)
        if diff:
            differences.append(diff)

    # Classify conflict type
    conflict_type = _classify_conflict(differences, claim_a, claim_b)

    # Generate summary
    summary = _generate_summary(
        claim_a, claim_b, differences, conflict_type, nli_result
    )

    return ContextComparisonResult(
        claim_a_id=claim_a.chunk_id,
        claim_b_id=claim_b.chunk_id,
        nli_label=nli_result.label,
        nli_confidence=nli_result.confidence,
        differences=differences,
        conflict_type=conflict_type,
        summary=summary,
    )


def _compare_dimension(
    dimension: str,
    val_a: str,
    val_b: str,
    base_description: str,
) -> Optional[ContextualDifference]:
    """
    Compare a single contextual dimension between two claims.

    Returns a ContextualDifference if a meaningful difference is detected,
    or None if the values are too similar or both are "Not reported".
    """
    # If both are missing, no useful comparison
    if val_a == "Not reported" and val_b == "Not reported":
        return None

    # If one is missing, note it
    if val_a == "Not reported" or val_b == "Not reported":
        known = val_b if val_a == "Not reported" else val_a
        return ContextualDifference(
            dimension=dimension,
            claim_a_value=val_a,
            claim_b_value=val_b,
            description=(
                f"One study does not report {dimension} "
                f"(the other reports: '{known}')"
            ),
        )

    # Check for meaningful difference
    if _texts_meaningfully_differ(val_a, val_b):
        return ContextualDifference(
            dimension=dimension,
            claim_a_value=val_a,
            claim_b_value=val_b,
            description=f"{base_description}: '{val_a}' vs '{val_b}'",
        )

    return None


def _texts_meaningfully_differ(a: str, b: str) -> bool:
    """
    Check if two text values are meaningfully different.

    Checks:
    1. Exact match (case-insensitive)
    2. Explicit clinical negation/subgroup divergence (e.g., 'obese' vs 'non-obese', 'with' vs 'without')
    3. Jaccard word similarity threshold (<0.6 treated as different)
    """
    a_str = a.lower().strip()
    b_str = b.lower().strip()
    if a_str == b_str:
        return False

    # Check for explicit clinical subgroup polarity
    negation_prefixes = ["non-", "non ", "not ", "without ", "no "]
    a_neg = any(neg in a_str for neg in negation_prefixes)
    b_neg = any(neg in b_str for neg in negation_prefixes)
    if a_neg != b_neg:
        return True

    words_a = set(re.findall(r"\b\w+\b", a_str))
    words_b = set(re.findall(r"\b\w+\b", b_str))
    if not words_a or not words_b:
        return True
    overlap = len(words_a & words_b)
    max_len = max(len(words_a), len(words_b))
    similarity = overlap / max_len
    return similarity < 0.6


def _classify_conflict(
    differences: List[ContextualDifference],
    claim_a: StructuredClaim,
    claim_b: StructuredClaim,
) -> str:
    """
    Classify the type of conflict based on identified differences.

    CONTEXTUAL: Meaningful differences in population, intervention, outcome, or duration
    SEMANTIC: Contradictory claims with very similar context
    UNRESOLVED: Insufficient information to classify
    """
    if not differences:
        # No contextual differences detected — claims appear to be in similar context
        if (claim_a.intervention != "Not reported" and
                claim_b.intervention != "Not reported"):
            return "SEMANTIC"
        return "UNRESOLVED"

    # Key dimensions that suggest contextual rather than semantic conflict
    key_contextual_dims = {"population", "intervention", "outcome", "duration"}
    contextual_dims_found = {d.dimension for d in differences}

    if key_contextual_dims & contextual_dims_found:
        return "CONTEXTUAL"

    return "UNRESOLVED"


def _generate_summary(
    claim_a: StructuredClaim,
    claim_b: StructuredClaim,
    differences: List[ContextualDifference],
    conflict_type: str,
    nli_result: NLIResult,
) -> str:
    """Generate a human-readable summary of the context comparison."""
    lines = []

    lines.append(
        f"NLI detected a potential {nli_result.label.lower()} "
        f"(confidence: {nli_result.confidence:.2f}) between:"
    )
    lines.append(f"  Claim A [{claim_a.chunk_id}]: {claim_a.to_claim_string()}")
    lines.append(f"  Claim B [{claim_b.chunk_id}]: {claim_b.to_claim_string()}")

    if differences:
        lines.append(f"\nIdentified contextual differences ({len(differences)}):")
        for diff in differences:
            lines.append(f"  • [{diff.dimension.upper()}] {diff.description}")
    else:
        lines.append("\nNo clear contextual differences detected.")

    if conflict_type == "CONTEXTUAL":
        lines.append(
            "\nNote: The apparent conflict may be related to differences in study context. "
            "This does NOT confirm that the studies are actually contradictory "
            "on the same clinical question."
        )
    elif conflict_type == "SEMANTIC":
        lines.append(
            "\nNote: The claims appear to address similar contexts but assert opposite directions. "
            "This represents a potential genuine scientific conflict requiring expert interpretation."
        )
    else:
        lines.append(
            "\nNote: Insufficient context to classify the nature of this conflict."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyze_contradictions(
    evidence_claims: List[StructuredClaim],
    nli_results: List[NLIResult],
    reference_claim: Optional[StructuredClaim] = None,
) -> List[ContextComparisonResult]:
    """
    Analyze all contradiction-labeled NLI results.

    For each CONTRADICTION:
      - If reference_claim is provided, compares the contradicting claim against it.
      - If reference_claim is None, searches for supporting claims in the pool to compare
        against pairwise (Study A vs Study B contradiction).
      - If no supporting claims exist, compares pairwise against the first evidence claim
        or analyzes internal clinical qualifiers.

    Returns:
        List of ContextComparisonResult for each detected contradiction.
    """
    contradiction_analyses: List[ContextComparisonResult] = []

    # Pair claims with their NLI results
    claim_nli_pairs = list(zip(evidence_claims, nli_results))
    contradictions = [p for p in claim_nli_pairs if p[1].label == "CONTRADICTION"]
    supporting = [p for p in claim_nli_pairs if p[1].label == "ENTAILMENT"]

    logger.info(f"Analyzing {len(contradictions)} detected contradictions...")

    for claim_b, nli in contradictions:
        if reference_claim:
            # Compare contradiction against the reference (query) claim
            analysis = compare_claim_context(reference_claim, claim_b, nli)
        elif supporting:
            # Compare contradiction pairwise against the strongest supporting claim (Study A vs Study B)
            best_supporting_claim = max(supporting, key=lambda x: x[1].confidence)[0]
            analysis = compare_claim_context(best_supporting_claim, claim_b, nli)
        elif len(evidence_claims) > 1:
            # Compare against the first non-identical evidence claim
            other_claim = next((c for c in evidence_claims if c.chunk_id != claim_b.chunk_id), evidence_claims[0])
            analysis = compare_claim_context(other_claim, claim_b, nli)
        else:
            # Fallback when single claim without reference
            analysis = ContextComparisonResult(
                claim_a_id="query",
                claim_b_id=claim_b.chunk_id,
                nli_label=nli.label,
                nli_confidence=nli.confidence,
                conflict_type="SEMANTIC",
                summary=(
                    f"Contradiction detected (confidence: {nli.confidence:.2f}). "
                    f"Evidence asserts opposite direction without detected contextual divergence."
                ),
            )
        contradiction_analyses.append(analysis)

    return contradiction_analyses
