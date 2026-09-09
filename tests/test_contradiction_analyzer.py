# =============================================================================
# tests/test_contradiction_analyzer.py
#
# Tests for Module 14 Contextual Contradiction Analyzer adhering to
# SRS Section 3.14 Acceptance Criteria.
# =============================================================================

import pytest
from src.claims.claim_extractor import StructuredClaim
from src.nli.nli_classifier import NLIResult
from src.context.contradiction_analyzer import (
    analyze_contradictions,
    ContextComparisonResult,
)


def test_srs_acceptance_criteria_1_contextual_difference():
    """
    SRS Section 3.14 Acceptance Criteria 1:
    GIVEN two claims with opposite conclusions (reduces vs does not reduce)
    but different patient populations (obese vs non-obese),
    WHEN contradiction analysis is performed,
    THEN the system SHALL classify the relationship as Contextual Difference (CONTEXTUAL),
    identify population as the discriminating dimension,
    and generate an explanation describing the population difference.
    """
    reference_claim = StructuredClaim(
        chunk_id="chunk_ukpds_obese",
        paper_id="ukpds_34",
        section="Results",
        intervention="metformin",
        population="overweight and obese type 2 diabetes",
        outcome="myocardial infarction mortality",
        direction="Reduction",
        raw_text="In overweight patients with type 2 diabetes, metformin was associated with significant reductions in MI.",
        study_context="Randomized Controlled Trial, n=1704",
    )

    contradicting_claim = StructuredClaim(
        chunk_id="chunk_non_obese",
        paper_id="non_obese_study",
        section="Results",
        intervention="metformin",
        population="normal-weight non-obese patients with type 2 diabetes",
        outcome="myocardial infarction mortality",
        direction="No significant change",
        raw_text="Metformin failed to show cardiovascular mortality benefit in normal-weight patients.",
        study_context="Randomized Controlled Trial, n=800",
    )

    nli_result = NLIResult(
        premise_chunk_id="chunk_non_obese",
        hypothesis_text="Metformin reduces cardiovascular mortality",
        label="CONTRADICTION",
        confidence=0.91,
        all_scores={"CONTRADICTION": 0.91, "ENTAILMENT": 0.04, "NEUTRAL": 0.05},
        premise_text=contradicting_claim.raw_text,
    )

    results = analyze_contradictions(
        evidence_claims=[contradicting_claim],
        nli_results=[nli_result],
        reference_claim=reference_claim,
    )

    assert len(results) == 1
    analysis = results[0]
    assert analysis.conflict_type == "CONTEXTUAL"
    assert any(d.dimension == "population" for d in analysis.differences)
    assert len(analysis.summary) > 0
    assert "population" in analysis.summary.lower()


def test_srs_acceptance_criteria_2_true_contradiction():
    """
    SRS Section 3.14 Acceptance Criteria 2:
    GIVEN two claims with opposite conclusions in identical populations and study designs,
    WHEN contradiction analysis is performed,
    THEN the system SHALL classify the relationship as True Contradiction (SEMANTIC)
    and identify the source of discrepancy.
    """
    reference_claim = StructuredClaim(
        chunk_id="ref_claim_01",
        paper_id="trial_a",
        section="Results",
        intervention="Drug X",
        population="type 2 diabetes",
        outcome="cardiovascular events",
        direction="Reduction",
        raw_text="Drug X significantly lowered major adverse cardiovascular events in T2D.",
        study_context="Randomized Controlled Trial",
    )

    opposing_claim = StructuredClaim(
        chunk_id="opp_claim_01",
        paper_id="trial_b",
        section="Results",
        intervention="Drug X",
        population="type 2 diabetes",
        outcome="cardiovascular events",
        direction="No significant change",
        raw_text="Drug X showed no reduction in major adverse cardiovascular events in T2D.",
        study_context="Randomized Controlled Trial",
    )

    nli_result = NLIResult(
        premise_chunk_id="opp_claim_01",
        hypothesis_text="Drug X reduces cardiovascular events in T2D",
        label="CONTRADICTION",
        confidence=0.89,
        all_scores={"CONTRADICTION": 0.89, "ENTAILMENT": 0.03, "NEUTRAL": 0.08},
        premise_text=opposing_claim.raw_text,
    )

    results = analyze_contradictions(
        evidence_claims=[opposing_claim],
        nli_results=[nli_result],
        reference_claim=reference_claim,
    )

    assert len(results) == 1
    analysis = results[0]
    assert analysis.conflict_type == "SEMANTIC"
    assert len(analysis.summary) > 0
