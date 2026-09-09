# =============================================================================
# tests/test_grade_status.py
#
# Tests for Module 15 GRADE/ADA Grader & Consensus Evidence Status Engine
# adhering to SRS Section 3.15 Acceptance Criteria.
# =============================================================================

import pytest
from src.claims.claim_extractor import StructuredClaim
from src.preprocessing.chunker import Chunk
from src.evidence.relationship import EvidenceItem, EvidenceRelationshipSummary
from src.evidence.grader import grade_grader, GRADEAssessment
from src.evidence.status import determine_evidence_status, SUPPORTED, INCONCLUSIVE


def _make_item(
    study_type: str,
    relationship: str,
    claim_text: str,
    sample_size: int = 1000,
    rob_phrase: str = "",
) -> EvidenceItem:
    chunk_id = f"chunk_{study_type[:3]}_{relationship[:3]}_{abs(hash(claim_text))}"
    full_text = f"{claim_text} n={sample_size}. {rob_phrase}"
    chunk = Chunk(
        chunk_id=chunk_id,
        paper_id=f"paper_{study_type[:3]}_{abs(hash(claim_text)) % 1000}",
        pmc_id="PMC12345",
        title=f"Study of Drug A in {study_type}",
        authors=["Author A"],
        year="2022",
        journal="Diabetes Care",
        doi="10.1234/dc",
        section="Results",
        chunk_index=0,
        study_type=study_type,
        source="pmc_xml",
        text=full_text,
        token_count=len(full_text.split()),
    )
    claim = StructuredClaim(
        chunk_id=chunk_id,
        paper_id=chunk.paper_id,
        section="Results",
        intervention="Drug A",
        outcome="Outcome B",
        direction="Reduction" if relationship == "SUPPORTING" else "No significant change",
        raw_text=full_text,
    )
    grade = grade_grader.grade_evidence(chunk=chunk, claim=claim)
    nli_label = "ENTAILMENT" if relationship == "SUPPORTING" else "CONTRADICTION"
    from src.nli.nli_classifier import NLIResult
    nli_res = NLIResult(
        premise_chunk_id=chunk.chunk_id,
        hypothesis_text="Drug A improves Outcome B",
        label=nli_label,
        confidence=0.90,
        all_scores={nli_label: 0.90},
        premise_text=full_text,
    )
    return EvidenceItem(
        chunk=chunk,
        claim=claim,
        nli_result=nli_res,
        relationship=relationship,
        grade=grade,
    )


def test_srs_acceptance_criteria_grade_weighting():
    """
    SRS Section 3.15 Acceptance Criteria:
    GIVEN a claim supported by 2 RCTs (high certainty)
    and contradicted by 8 observational studies (low certainty),
    WHEN evidence status is determined,
    THEN the status SHALL be SUPPORTED,
    and the confidence score SHALL reflect the higher weighting of the RCT evidence.
    """
    # 2 RCTs (Large sample n=2500 -> upgraded to HIGH tier, weight 4.0 each -> total support weight = 8.0)
    supporting_rcts = [
        _make_item(
            study_type="Randomized Controlled Trial",
            relationship="Supports",
            claim_text=f"RCT {i}: Drug A significantly reduced cardiovascular risk in randomized trial.",
            sample_size=2500,
        )
        for i in range(1, 3)
    ]

    # 8 Observational Studies (n=45, high risk of bias -> VERY LOW tier, weight 0.5 each -> total contradicting weight = 4.0)
    # Total Support Weight = 8.0, Contradiction Weight = 4.0 (2.0x ratio in favor of support)
    contradicting_observational = [
        _make_item(
            study_type="Observational Study",
            relationship="Contradicts",
            claim_text=f"Observational cohort {i}: Drug A was not associated with improved outcomes.",
            sample_size=45,
            rob_phrase="high risk of bias and lack of blinding.",
        )
        for i in range(1, 9)
    ]

    all_items = supporting_rcts + contradicting_observational
    from src.evidence.relationship import summarize_evidence_relationships
    summary = summarize_evidence_relationships(all_items)

    result = determine_evidence_status(
        summary=summary,
        avg_nli_confidence=0.88,
        avg_retrieval_relevance=0.82,
    )

    # Status must be SUPPORTED because high-certainty RCTs outvote low-certainty observational studies
    assert result.status == SUPPORTED
    assert result.weighted_support > result.weighted_contradiction
    assert "grade-weighted strength" in result.rationale.lower()

    # Verify 7-factor confidence breakdown
    breakdown = result.confidence_breakdown
    expected_factors = [
        "evidence_agreement",
        "evidence_quality_tier",
        "nli_certainty",
        "independent_study_count",
        "context_consistency",
        "citation_support",
        "contradiction_penalty",
    ]
    for factor in expected_factors:
        assert factor in breakdown
        assert 0.0 <= breakdown[factor] <= 1.0
