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
    Compare a single contextual dimension between two claims per FR-14.2/FR-14.4.

    Returns a ContextualDifference if an actual, material clinical difference
    is detected, or None if the values are comparable or missing.
    Missing metadata in one or both claims does NOT constitute a contextual difference.
    """
    if val_a == "Not reported" or val_b == "Not reported":
        return None

    # Compare based on specific dimension semantics
    if dimension == "population":
        return _compare_population(val_a, val_b, base_description)
    elif dimension == "intervention":
        return _compare_intervention(val_a, val_b, base_description)
    elif dimension == "comparator":
        return _compare_comparator(val_a, val_b, base_description)
    elif dimension == "outcome":
        return _compare_outcome(val_a, val_b, base_description)
    elif dimension == "duration":
        return _compare_duration(val_a, val_b, base_description)
    elif dimension == "study_context":
        # Per FR-14.4 and SRS 3.14: study design, sample size, and bias differences
        # are methodological certainty factors handled in Module 15 (GRADE),
        # not contextual differences in Module 14.
        return None

    return None


def _clean_text(t: str) -> str:
    """Normalize whitespace and lowercasing."""
    return re.sub(r"\s+", " ", t.lower().strip())


def _compare_population(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare patient populations for explicit, named subgroup divergences."""
    a = _clean_text(val_a)
    b = _clean_text(val_b)
    if a == b:
        return None

    # Check 1: Explicit clinical subgroup polarity / negation (e.g. obese vs non-obese, with vs without)
    polarity_contrasts = [
        (["non-obese", "non obese", "normal weight", "lean", "bmi < 25", "without obesity"],
         ["obese", "obesity", "overweight", "bmi >= 30", "bmi > 30"]),
        (["without ckd", "without kidney disease", "without renal impairment", "normal renal function", "preserved egfr"],
         ["with ckd", "chronic kidney disease", "renal impairment", "nephropathy", "esrd", "dialysis", "egfr < 30", "egfr < 60"]),
        (["without heart failure", "no heart failure", "without hf"],
         ["heart failure", "hfref", "hfpef", "congestive heart failure", "with heart failure"]),
        (["without cvd", "primary prevention", "no prior cv", "low cv risk"],
         ["established cvd", "secondary prevention", "prior mi", "coronary artery disease", "cad", "ascvd"]),
        (["pediatric", "children", "adolescents", "youth", "< 18"],
         ["adult", "adults", "elderly", "older adults", ">= 65"]),
        (["type 1 diabetes", "t1d", "t1dm"],
         ["type 2 diabetes", "t2d", "t2dm"]),
        (["gestational diabetes", "gdm"],
         ["type 2 diabetes", "t2d", "t2dm"]),
    ]

    for group1, group2 in polarity_contrasts:
        a_in_g1 = any(term in a for term in group1)
        b_in_g1 = any(term in b for term in group1)
        a_in_g2 = any(term in a for term in group2)
        b_in_g2 = any(term in b for term in group2)

        if (a_in_g1 and b_in_g2) or (a_in_g2 and b_in_g1):
            return ContextualDifference(
                dimension="population",
                claim_a_value=val_a,
                claim_b_value=val_b,
                description=f"Studies examined divergent patient cohorts: '{val_a}' vs '{val_b}'",
            )

    # Check 2: Explicit comorbidity restriction vs general T2D
    # e.g., one study restricted to severe ESRD/dialysis while the other is general T2D
    restricted_subgroups = [
        "esrd", "dialysis", "egfr < 30", "severe renal impairment",
        "heart failure with reduced ejection fraction", "hfref",
        "pediatric", "children", "pregnancy", "gestational",
    ]
    for sub in restricted_subgroups:
        a_has_sub = sub in a
        b_has_sub = sub in b
        if a_has_sub != b_has_sub:
            restricted_val = val_a if a_has_sub else val_b
            general_val = val_b if a_has_sub else val_a
            return ContextualDifference(
                dimension="population",
                claim_a_value=val_a,
                claim_b_value=val_b,
                description=f"Evidence population restricted to '{restricted_val}' subgroup vs general cohort '{general_val}'",
            )

    # If both describe general T2D (even with common descriptors like 'patients with type 2 diabetes and obesity'),
    # they are clinically comparable cohorts, not divergent contexts.
    return None


def _compare_intervention(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare interventions for material drug mismatch."""
    a = _clean_text(val_a)
    b = _clean_text(val_b)
    if a == b:
        return None

    # Normalize out generic medication words
    def _drug_stem(t: str) -> str:
        s = re.sub(r"\b(monotherapy|combination|therapy|treatment|oral|tablets?|hydrochloride|daily|dose)\b", "", t)
        return re.sub(r"\s+", " ", s).strip()

    stem_a = _drug_stem(a)
    stem_b = _drug_stem(b)
    if stem_a and stem_b and (stem_a in stem_b or stem_b in stem_a):
        return None

    # Check for different active ingredients
    return ContextualDifference(
        dimension="intervention",
        claim_a_value=val_a,
        claim_b_value=val_b,
        description=f"Studies examined different interventions: '{val_a}' vs '{val_b}'",
    )


def _compare_comparator(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare control/comparator for placebo vs active comparator differences."""
    a = _clean_text(val_a)
    b = _clean_text(val_b)
    if a == b:
        return None

    placebo_terms = ["placebo", "standard care", "usual care", "control", "standard of care"]
    a_is_placebo = any(p in a for p in placebo_terms)
    b_is_placebo = any(p in b for p in placebo_terms)

    if a_is_placebo != b_is_placebo:
        return ContextualDifference(
            dimension="comparator",
            claim_a_value=val_a,
            claim_b_value=val_b,
            description=f"Studies used different comparators: '{val_a}' vs '{val_b}'",
        )

    return None


def _compare_outcome(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare outcomes for surrogate biomarker vs hard clinical endpoint divergence."""
    a = _clean_text(val_a)
    b = _clean_text(val_b)
    if a == b:
        return None

    # Surrogate glycemic endpoints vs hard clinical endpoints
    surrogates = ["hba1c", "blood glucose", "glycemic control", "fpg", "postprandial glucose"]
    hard_endpoints = ["mortality", "death", "myocardial infarction", "stroke", "heart failure hospitalization", "mace"]

    a_is_surrogate = any(s in a for s in surrogates)
    b_is_surrogate = any(s in b for s in surrogates)
    a_is_hard = any(h in a for h in hard_endpoints)
    b_is_hard = any(h in b for h in hard_endpoints)

    if (a_is_surrogate and b_is_hard) or (a_is_hard and b_is_surrogate):
        return ContextualDifference(
            dimension="outcome",
            claim_a_value=val_a,
            claim_b_value=val_b,
            description=f"Studies measured different outcome tiers: surrogate endpoint ({val_a}) vs clinical endpoint ({val_b})",
        )

    # Both are within same broad outcome domain (e.g. CV mortality vs mortality vs MACE)
    if (a_is_hard and b_is_hard) or (a_is_surrogate and b_is_surrogate):
        return None

    # Check word overlap
    words_a = set(re.findall(r"\b\w{3,}\b", a))
    words_b = set(re.findall(r"\b\w{3,}\b", b))
    if words_a & words_b:
        return None

    return ContextualDifference(
        dimension="outcome",
        claim_a_value=val_a,
        claim_b_value=val_b,
        description=f"Studies measured different outcomes: '{val_a}' vs '{val_b}'",
    )


def _compare_duration(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare duration for acute vs chronic study design divergence."""
    a = _clean_text(val_a)
    b = _clean_text(val_b)
    if a == b:
        return None

    acute_terms = ["acute", "hours", "days", "single dose", "in-hospital", "1 week", "2 weeks"]
    chronic_terms = ["years", "long-term", "52 weeks", "104 weeks", "multi-year", "5 years"]

    a_acute = any(t in a for t in acute_terms)
    b_acute = any(t in b for t in acute_terms)
    a_chronic = any(t in a for t in chronic_terms)
    b_chronic = any(t in b for t in chronic_terms)

    if (a_acute and b_chronic) or (a_chronic and b_acute):
        return ContextualDifference(
            dimension="duration",
            claim_a_value=val_a,
            claim_b_value=val_b,
            description=f"Studies differed in timeframe: acute ({val_a}) vs chronic/long-term ({val_b})",
        )

    return None


def _classify_conflict(
    differences: List[ContextualDifference],
    claim_a: StructuredClaim,
    claim_b: StructuredClaim,
) -> str:
    """
    Classify the type of conflict based on identified differences per FR-14.3/FR-14.4.

    SEMANTIC (True Contradiction): Default when Intervention, Population (at stated
        specificity), and Outcome match and NLI indicates contradiction.
    CONTEXTUAL: Material, named clinical divergence (e.g. population subgroup
        polarity/divergence, active comparator vs placebo, acute vs chronic duration).
    """
    # If explicit, named clinical differences were identified, route to CONTEXTUAL
    clinical_diffs = [
        d for d in differences
        if d.dimension in ("population", "intervention", "comparator", "outcome", "duration")
    ]

    if clinical_diffs:
        return "CONTEXTUAL"

    # Default to SEMANTIC (True Contradiction) when PICO context aligns
    return "SEMANTIC"


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
