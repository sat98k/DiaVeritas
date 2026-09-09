# =============================================================================
# tests/test_phase2_verdict_gate.py
#
# Phase 2: Unit tests enforcing FR-16.2 / FR-16.3 minimum-evidence gate for verdicts.
# Conclusive verdicts (SUPPORTED/REFUTED) require at least 2 distinct independent studies.
# =============================================================================

import pytest
from src.claims.claim_extractor import StructuredClaim
from src.preprocessing.chunker import Chunk
from src.evidence.relationship import EvidenceItem, EvidenceRelationshipSummary
from src.evidence.grader import GRADEAssessment
from src.evidence.status import determine_evidence_status, SUPPORTED, REFUTED, INCONCLUSIVE
from src.nli.nli_classifier import NLIResult


def _create_study_item(
    paper_id: str,
    chunk_id: str,
    relationship: str,
    grade_weight: float = 4.0,
    confidence: float = 0.94,
) -> EvidenceItem:
    chunk = Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        pmc_id="PMC99999",
        title="Study Title",
        authors=["Author et al."],
        year="2022",
        journal="NEJM",
        doi="10.1056/nejm12345",
        section="Results",
        chunk_index=0,
        study_type="Randomized Controlled Trial",
        source="pmc_xml",
        text="Empagliflozin reduced heart failure hospitalization in patients with type 2 diabetes.",
        token_count=100,
    )
    claim = StructuredClaim(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Results",
        intervention="Empagliflozin",
        outcome="Heart Failure Hospitalization",
        direction="Reduction" if relationship == "Supports" else "No change",
        raw_text="Empagliflozin reduced heart failure hospitalization in patients with type 2 diabetes.",
    )
    grade = GRADEAssessment(
        chunk_id=chunk_id,
        paper_id=paper_id,
        tier="HIGH",
        weight=grade_weight,
        study_design="Randomized Controlled Trial",
        risk_of_bias="Low",
        sample_size_note="n=7020",
        rationale="Landmark RCT (EMPA-REG OUTCOME)",
    )
    nli_label = "ENTAILMENT" if relationship == "Supports" else "CONTRADICTION"
    nli = NLIResult(
        premise_chunk_id=chunk_id,
        hypothesis_text="Empagliflozin reduces heart failure hospitalization",
        label=nli_label,
        confidence=confidence,
        all_scores={nli_label: confidence},
        premise_text=chunk.text,
    )
    return EvidenceItem(
        chunk=chunk,
        claim=claim,
        nli_result=nli,
        relationship=relationship,
        grade=grade,
    )


def test_q2c_single_supporting_study_forces_inconclusive():
    """
    Q2C scenario:
    1 Supporting study (EMPA-REG OUTCOME, GRADE 4.0, NLI confidence 0.94), 0 contradictions.
    Under FR-16.2, a single study cannot issue a conclusive SUPPORTED verdict.
    Must force INCONCLUSIVE and cap confidence breakdown.
    """
    item1 = _create_study_item(
        paper_id="paper_empareg_2015",
        chunk_id="chunk_empareg_01",
        relationship="Supports",
        grade_weight=4.0,
        confidence=0.94,
    )

    summary = EvidenceRelationshipSummary(
        total=1,
        supporting=[item1],
        contradicting=[],
        contextual=[],
        neutral=[],
    )

    result = determine_evidence_status(summary, avg_nli_confidence=0.94, avg_retrieval_relevance=0.90)

    # Acceptance gate: single study forces INCONCLUSIVE
    assert result.status == INCONCLUSIVE, f"Expected INCONCLUSIVE for 1 study, got {result.status}"
    assert "Insufficient study replication" in result.rationale
    assert result.confidence_breakdown["independent_study_count"] == 0.25


def test_multiple_chunks_from_same_paper_counts_as_one_study():
    """
    If 3 chunks are retrieved but all come from the SAME paper_id (e.g. Results, Discussion, Abstract),
    the system must recognize distinct study count == 1 and force INCONCLUSIVE (FR-16.2).
    """
    items = [
        _create_study_item(
            paper_id="paper_same_study_001",
            chunk_id=f"chunk_same_{i}",
            relationship="Supports",
            grade_weight=4.0,
            confidence=0.92,
        )
        for i in range(1, 4)
    ]

    summary = EvidenceRelationshipSummary(
        total=3,
        supporting=items,
        contradicting=[],
        contextual=[],
        neutral=[],
    )

    result = determine_evidence_status(summary, avg_nli_confidence=0.92, avg_retrieval_relevance=0.88)

    assert result.status == INCONCLUSIVE
    assert "only 1 independent study" in result.rationale


def test_two_distinct_independent_studies_achieve_supported():
    """
    When 2 distinct high-strength studies (e.g. EMPA-REG and DAPA-HF with different paper_ids)
    support the hypothesis with 0 contradictions, SUPPORTED must be reachable.
    """
    item1 = _create_study_item(
        paper_id="paper_empareg_2015",
        chunk_id="chunk_empareg_01",
        relationship="Supports",
        grade_weight=4.0,
        confidence=0.94,
    )
    item2 = _create_study_item(
        paper_id="paper_dapahf_2019",
        chunk_id="chunk_dapahf_01",
        relationship="Supports",
        grade_weight=4.0,
        confidence=0.91,
    )

    summary = EvidenceRelationshipSummary(
        total=2,
        supporting=[item1, item2],
        contradicting=[],
        contextual=[],
        neutral=[],
    )

    result = determine_evidence_status(summary, avg_nli_confidence=0.92, avg_retrieval_relevance=0.89)

    assert result.status == SUPPORTED
    assert "2 independent studies" in result.rationale
    assert result.confidence_breakdown["independent_study_count"] == 0.65


def test_single_contradicting_study_forces_inconclusive():
    """
    Symmetric gate for REFUTED (FR-16.3): single contradicting study must not issue REFUTED.
    """
    con_item = _create_study_item(
        paper_id="paper_single_con",
        chunk_id="chunk_con_01",
        relationship="Contradicts",
        grade_weight=3.5,
        confidence=0.88,
    )

    summary = EvidenceRelationshipSummary(
        total=1,
        supporting=[],
        contradicting=[con_item],
        contextual=[],
        neutral=[],
    )

    result = determine_evidence_status(summary, avg_nli_confidence=0.88, avg_retrieval_relevance=0.80)

    assert result.status == INCONCLUSIVE
    assert "Insufficient study replication" in result.rationale
