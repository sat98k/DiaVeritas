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
from typing import List, Dict, Any, Optional, Tuple, Set
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


_PLACEHOLDER_METADATA_VALUES = {
    "", "not reported", "not specified", "unspecified", "none", "unknown",
    "target intervention", "target outcome", "target population", "target comparator",
    "unspecified endpoint", "general t2d care", "treatment",
}


def _is_missing_metadata(val: Optional[str]) -> bool:
    """Check whether a field value represents missing or placeholder metadata."""
    if val is None:
        return True
    return str(val).strip().lower() in _PLACEHOLDER_METADATA_VALUES


def _clean_text(t: str) -> str:
    """Normalize whitespace and lowercasing."""
    return re.sub(r"\s+", " ", t.lower().strip())


def _interventions_match(int_a: str, int_b: str, text_a: str = "", text_b: str = "") -> bool:
    """
    Generic biomedical intervention matching.
    Handles exact names, drug stems, drug classes/members, lifestyle synonyms, and text containment.
    """
    if _is_missing_metadata(int_a) or _is_missing_metadata(int_b):
        known = int_a if not _is_missing_metadata(int_a) else int_b
        target_text = text_b if not _is_missing_metadata(int_a) else text_a
        if known and target_text:
            k_low = known.lower()
            t_low = target_text.lower()
            if re.search(r"\b" + re.escape(k_low) + r"\b", t_low):
                return True
            from src.claims.query_normalizer import DRUG_CLASSES, DRUG_BRAND_MAP
            for class_id, info in DRUG_CLASSES.items():
                all_terms = set(info["class_terms"] + info["members"])
                for member in info["members"]:
                    for brand in DRUG_BRAND_MAP.get(member, []):
                        all_terms.add(brand)
                if any(re.search(r"\b" + re.escape(term) + r"\b", k_low) for term in all_terms):
                    if any(re.search(r"\b" + re.escape(term) + r"\b", t_low) for term in all_terms):
                        return True
            lifestyle_clusters = {
                "aerobic_exercise": ["aerobic exercise", "aerobic training", "walking", "running", "cycling", "endurance training"],
                "resistance_training": ["resistance training", "strength training", "weight training", "resistance exercise"],
                "diet": ["diet", "dietary intervention", "nutrition", "caloric restriction", "mediterranean diet"],
            }
            for cluster_id, terms in lifestyle_clusters.items():
                if any(t in k_low for t in terms):
                    if any(re.search(r"\b" + re.escape(t) + r"\b", t_low) for t in terms):
                        return True
            general_exercise = ["exercise", "physical activity", "exercise/physical activity"]
            if any(k_low == g or k_low.strip() == g for g in general_exercise):
                all_ex = [t for terms in [lifestyle_clusters["aerobic_exercise"], lifestyle_clusters["resistance_training"]] for t in terms]
                if any(re.search(r"\b" + re.escape(t) + r"\b", t_low) for t in all_ex):
                    return True
            return False
        return True

    a = _clean_text(int_a)
    b = _clean_text(int_b)
    if a == b:
        return True

    # Drug stem comparison
    def _drug_stem(t: str) -> str:
        s = re.sub(r"\b(monotherapy|combination|therapy|treatment|oral|tablets?|hydrochloride|daily|dose|inhibitors?|agonists?)\b", "", t)
        return re.sub(r"\s+", " ", s).strip()

    stem_a = _drug_stem(a)
    stem_b = _drug_stem(b)
    if stem_a and stem_b and (stem_a == stem_b or stem_a in stem_b or stem_b in stem_a):
        return True

    # Drug class to member matching
    from src.claims.query_normalizer import DRUG_CLASSES, DRUG_BRAND_MAP
    for class_id, info in DRUG_CLASSES.items():
        all_terms = set(info["class_terms"] + info["members"])
        for member in info["members"]:
            for brand in DRUG_BRAND_MAP.get(member, []):
                all_terms.add(brand)

        a_matches = any(re.search(r"\b" + re.escape(term) + r"\b", a) for term in all_terms)
        b_matches = any(re.search(r"\b" + re.escape(term) + r"\b", b) for term in all_terms)
        if a_matches and b_matches:
            return True

    # Lifestyle interventions matching
    lifestyle_clusters = {
        "aerobic_exercise": ["aerobic exercise", "aerobic training", "walking", "running", "cycling", "endurance training"],
        "resistance_training": ["resistance training", "strength training", "weight training", "resistance exercise"],
        "diet": ["diet", "dietary intervention", "nutrition", "caloric restriction", "mediterranean diet"],
    }
    for cluster_id, terms in lifestyle_clusters.items():
        a_in_cluster = any(t in a for t in terms)
        b_in_cluster = any(t in b for t in terms)
        if a_in_cluster and b_in_cluster:
            return True

    # General exercise match: only standalone "exercise" or "physical activity" matches specific exercise types
    general_exercise = ["exercise", "physical activity", "exercise/physical activity"]
    if any(a == g or a.strip() == g for g in general_exercise) and any(t in b for terms in [lifestyle_clusters["aerobic_exercise"], lifestyle_clusters["resistance_training"]] for t in terms):
        return True
    if any(b == g or b.strip() == g for g in general_exercise) and any(t in a for terms in [lifestyle_clusters["aerobic_exercise"], lifestyle_clusters["resistance_training"]] for t in terms):
        return True

    # Text containment fallback
    if (text_b and re.search(r"\b" + re.escape(a) + r"\b", text_b.lower())) or \
       (text_a and re.search(r"\b" + re.escape(b) + r"\b", text_a.lower())):
        return True

    return False


def _outcomes_match(out_a: str, out_b: str, text_a: str = "", text_b: str = "") -> bool:
    """
    Generic biomedical outcome matching.
    Handles exact names, clinical synonym clusters, substrings, and word overlap.
    """
    if _is_missing_metadata(out_a) or _is_missing_metadata(out_b):
        return True

    a = _clean_text(out_a)
    b = _clean_text(out_b)
    if a == b:
        return True

    from src.claims.query_normalizer import OUTCOME_SYNONYM_CLUSTERS
    for cluster_id, synonyms in OUTCOME_SYNONYM_CLUSTERS.items():
        a_in_cluster = any(s == a or s in a or a in s for s in synonyms)
        b_in_cluster = any(s in b or b in s for s in synonyms)
        if a_in_cluster and b_in_cluster:
            return True

    # Cross-cluster clinical equivalence: CV mortality matches mortality and cv_events
    cv_mortality_terms = ["cardiovascular mortality", "cardiovascular death", "cv death", "fatal cv events"]
    if any(t in a for t in cv_mortality_terms) and any(s in b for s in OUTCOME_SYNONYM_CLUSTERS.get("mortality", [])):
        return True
    if any(t in b for t in cv_mortality_terms) and any(s in a for s in OUTCOME_SYNONYM_CLUSTERS.get("mortality", [])):
        return True

    # Substring match
    if a in b or b in a:
        return True

    # Text containment fallback
    if (text_b and re.search(r"\b" + re.escape(a) + r"\b", text_b.lower())) or \
       (text_a and re.search(r"\b" + re.escape(b) + r"\b", text_a.lower())):
        return True

    # Significant biomedical token overlap
    words_a = set(re.findall(r"\b[a-z]{4,}\b", a))
    words_b = set(re.findall(r"\b[a-z]{4,}\b", b))
    overlap = words_a & words_b - {"events", "event", "risk", "level", "levels", "control", "status"}
    if overlap:
        return True

    return False


def _populations_compatible(pop_a: str, pop_b: str) -> Tuple[bool, Optional[str]]:
    """
    Determine whether two populations are clinically compatible.
    Returns (True, None) if compatible, or (False, reason) if mutually exclusive/divergent.
    """
    diff = _compare_population(pop_a, pop_b, "")
    if diff:
        return False, diff.description
    return True, None


def _normalize_direction(direction_str: Optional[str], text: str = "") -> str:
    """Normalize effect direction to 'REDUCTION', 'INCREASE', 'NO_CHANGE', or 'UNCLEAR'."""
    d = (direction_str or "").strip().lower()
    if d in ("reduction", "decrease", "lower", "loss", "decline", "drop", "prevent", "lessen", "fall"):
        return "REDUCTION"
    if d in ("increase", "elevation", "elevate", "raise", "higher", "gain", "rise", "grow", "accrue"):
        return "INCREASE"
    if d in ("no change", "neutral", "similar", "unchanged", "no difference", "not significant"):
        return "NO_CHANGE"

    # Fallback to inspecting text
    t = text.lower()
    if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in [
        "reduce", "reduces", "reduced", "reducing", "reduction",
        "decrease", "decreases", "decreased", "decreasing",
        "lower", "lowers", "lowered", "lowering",
        "prevent", "prevents", "prevention"
    ]):
        return "REDUCTION"
    if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in [
        "increase", "increases", "increased", "increasing",
        "elevate", "elevates", "elevated", "elevating",
        "raise", "raises", "raised", "raising"
    ]):
        return "INCREASE"
    if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in ["no difference", "no significant difference", "did not differ"]):
        return "NO_CHANGE"

    return "UNCLEAR"


def _is_opposite_direction(dir_a: str, dir_b: str) -> bool:
    """Check if two directions represent direct opposite clinical effects."""
    return (dir_a == "REDUCTION" and dir_b == "INCREASE") or (dir_a == "INCREASE" and dir_b == "REDUCTION")


def _is_matching_direction(dir_a: str, dir_b: str) -> bool:
    """Check if two directions represent matching clinical effects."""
    return (dir_a == "REDUCTION" and dir_b == "REDUCTION") or (dir_a == "INCREASE" and dir_b == "INCREASE")


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
    if _is_missing_metadata(val_a) or _is_missing_metadata(val_b):
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
        return None

    return None


def _compare_population(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare patient populations for explicit, named subgroup divergences."""
    if _is_missing_metadata(val_a) or _is_missing_metadata(val_b):
        return None

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

    return None


def _compare_intervention(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare interventions for material drug mismatch."""
    if _is_missing_metadata(val_a) or _is_missing_metadata(val_b):
        return None

    if _interventions_match(val_a, val_b):
        return None

    return ContextualDifference(
        dimension="intervention",
        claim_a_value=val_a,
        claim_b_value=val_b,
        description=f"Studies examined different interventions: '{val_a}' vs '{val_b}'",
    )


def _compare_comparator(val_a: str, val_b: str, base_desc: str) -> Optional[ContextualDifference]:
    """Compare control/comparator for placebo vs active comparator differences."""
    if _is_missing_metadata(val_a) or _is_missing_metadata(val_b):
        return None

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
    if _is_missing_metadata(val_a) or _is_missing_metadata(val_b):
        return None

    if _outcomes_match(val_a, val_b):
        return None

    a = _clean_text(val_a)
    b = _clean_text(val_b)

    # Surrogate glycemic endpoints vs hard clinical endpoints
    surrogates = ["hba1c", "blood glucose", "glycemic control", "fpg", "postprandial glucose"]
    hard_endpoints = ["mortality", "death", "myocardial infarction", "stroke", "heart failure hospitalization", "mace", "heart failure"]

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
    if _is_missing_metadata(val_a) or _is_missing_metadata(val_b):
        return None

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
    Classify the type of conflict based on identified differences per FR-14.3/FR-14.4 and General Polarity Rule.

    GENERAL RULE:
    If:
      - intervention matches,
      - outcome matches,
      - population is compatible,
      - and evidence effect direction is opposite to query hypothesis (or claims assert opposite directions)
    then:
      → SEMANTIC (True Contradiction)

    If population is divergent (e.g. pediatric vs adult), or intervention/outcome is divergent:
      → CONTEXTUAL
    """
    # 1. Check for explicit population divergence (e.g. pediatric vs adult)
    pop_compatible, _ = _populations_compatible(claim_a.population, claim_b.population)
    if not pop_compatible:
        return "CONTEXTUAL"

    # 2. Check for explicit intervention divergence
    if not _interventions_match(claim_a.intervention, claim_b.intervention, claim_a.raw_text, claim_b.raw_text):
        return "CONTEXTUAL"

    # 3. Check for explicit outcome tier divergence
    if not _outcomes_match(claim_a.outcome, claim_b.outcome, claim_a.raw_text, claim_b.raw_text):
        return "CONTEXTUAL"

    # 4. Check directions
    dir_a = _normalize_direction(claim_a.direction, claim_a.raw_text)
    dir_b = _normalize_direction(claim_b.direction, claim_b.raw_text)

    # If intervention matches, outcome matches, population is compatible,
    # and directions are opposite -> SEMANTIC (True Contradiction)
    if _is_opposite_direction(dir_a, dir_b):
        return "SEMANTIC"

    # If differences contain an explicit population, intervention, or outcome difference:
    major_diffs = [
        d for d in differences
        if d.dimension in ("population", "intervention", "outcome")
    ]
    if major_diffs:
        return "CONTEXTUAL"

    # Default to SEMANTIC when PICO context aligns
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
