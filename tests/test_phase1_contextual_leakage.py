# =============================================================================
# tests/test_phase1_contextual_leakage.py
#
# Phase 1: Unit tests verifying contextual differences are decoupled from
# contradiction weighting (FR-14.3, FR-14.4, FR-16.1).
# =============================================================================

import pytest
from src.claims.claim_extractor import StructuredClaim
from src.preprocessing.chunker import Chunk
from src.evidence.relationship import EvidenceItem, EvidenceRelationshipSummary
from src.evidence.grader import GRADEAssessment
from src.evidence.status import determine_evidence_status, SUPPORTED, INCONCLUSIVE
from src.nli.nli_classifier import NLIResult
from src.context.contradiction_analyzer import ContextComparisonResult, ContextualDifference


def _create_item(
    chunk_id: str,
    paper_id: str,
    relationship: str,
    grade_weight: float = 4.0,
    context_diffs: list = None,
) -> EvidenceItem:
    chunk = Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        pmc_id="PMC12345",
        title="Clinical Trial",
        authors=["Author et al."],
        year="2022",
        journal="Diabetes Care",
        doi="10.1234/dc",
        section="Results",
        chunk_index=0,
        study_type="Randomized Controlled Trial",
        source="pmc_xml",
        text="Clinical trial evidence text.",
        token_count=100,
    )
    claim = StructuredClaim(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Results",
        intervention="Dapagliflozin",
        outcome="Heart Failure Hospitalization",
        direction="Reduction" if relationship == "Supports" else "No change",
        raw_text="Clinical trial evidence text.",
    )
    grade = GRADEAssessment(
        chunk_id=chunk_id,
        paper_id=paper_id,
        tier="HIGH",
        weight=grade_weight,
        study_design="Randomized Controlled Trial",
        risk_of_bias="Low",
        sample_size_note="n=4744",
        rationale="High certainty RCT",
    )
    nli_label = "ENTAILMENT" if relationship == "Supports" else ("CONTRADICTION" if relationship == "Contradicts" else "NEUTRAL")
    nli = NLIResult(
        premise_chunk_id=chunk_id,
        hypothesis_text="Dapagliflozin reduces HHF",
        label=nli_label,
        confidence=0.92,
        all_scores={nli_label: 0.92},
        premise_text="Clinical trial evidence text.",
    )
    ctx_analysis = None
    if context_diffs:
        ctx_analysis = ContextComparisonResult(
            claim_a_id="query",
            claim_b_id=chunk_id,
            nli_label="CONTRADICTION",
            nli_confidence=0.85,
            differences=[ContextualDifference(dimension=d[0], claim_a_value=d[1], claim_b_value=d[2], description=d[3]) for d in context_diffs],
            conflict_type="CONTEXTUAL",
            summary="Contextual difference in population",
        )

    return EvidenceItem(
        chunk=chunk,
        claim=claim,
        nli_result=nli,
        relationship=relationship,
        context_analysis=ctx_analysis,
        grade=grade,
    )


def test_q2d_contextual_difference_does_not_leak_into_contradiction_weight():
    """
    Q2D scenario:
    1 Supporting (GRADE weight 4.0), 0 Contradicting, 1 Contextual Difference (GRADE weight 3.0).
    Contradicting=0 MUST mean contradiction_weight == 0.0.
    """
    sup_item = _create_item(
        chunk_id="chunk_dapa_01",
        paper_id="paper_dapa_hf",
        relationship="Supports",
        grade_weight=4.0,
    )
    ctx_item = _create_item(
        chunk_id="chunk_dapa_ctx_02",
        paper_id="paper_dapa_subgroup",
        relationship="Contextual Difference",
        grade_weight=3.0,
        context_diffs=[("population", "T2D with CKD", "T2D without CKD", "Population divergence")],
    )

    summary = EvidenceRelationshipSummary(
        total=2,
        supporting=[sup_item],
        contradicting=[],
        contextual=[ctx_item],
        neutral=[],
    )

    result = determine_evidence_status(summary, avg_nli_confidence=0.90, avg_retrieval_relevance=0.85)

    # Acceptance requirement: Contradicting=0 must produce contradiction_weight == 0.0
    assert result.n_contradicting == 0
    assert result.weighted_contradiction == 0.0, f"Expected 0.0 contradiction weight, got {result.weighted_contradiction}"
    assert result.contextual_weight == 3.0, f"Expected 3.0 contextual weight, got {result.contextual_weight}"


def test_true_contradiction_does_contribute_to_contradiction_weight():
    """
    Genuine True Contradiction case:
    Must contribute to weighted_contradiction.
    """
    con_item = _create_item(
        chunk_id="chunk_true_con_01",
        paper_id="paper_con",
        relationship="Contradicts",
        grade_weight=3.0,
    )

    summary = EvidenceRelationshipSummary(
        total=1,
        supporting=[],
        contradicting=[con_item],
        contextual=[],
        neutral=[],
    )

    result = determine_evidence_status(summary, avg_nli_confidence=0.88, avg_retrieval_relevance=0.80)

    assert result.n_contradicting == 1
    assert result.weighted_contradiction == 3.0, f"Expected 3.0 contradiction weight, got {result.weighted_contradiction}"
    assert result.contextual_weight == 0.0
