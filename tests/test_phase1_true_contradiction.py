# =============================================================================
# tests/test_phase1_true_contradiction.py
#
# Phase 1 Unit Tests: True Contradiction vs Contextual Difference (FR-14.3, FR-14.4)
# and End-to-End REFUTED Verdict Reachability (FR-16.3).
# =============================================================================

import pytest
from src.claims.claim_extractor import StructuredClaim
from src.preprocessing.chunker import Chunk
from src.nli.nli_classifier import NLIResult
from src.evidence.grader import GRADEAssessment
from src.context.contradiction_analyzer import analyze_contradictions
from src.evidence.relationship import (
    build_evidence_items,
    summarize_evidence_relationships,
)
from src.evidence.status import (
    determine_evidence_status,
    REFUTED,
    INCONCLUSIVE,
)


def _make_test_chunk(chunk_id: str, paper_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        pmc_id=f"PMC_{chunk_id}",
        title=f"Clinical Trial {paper_id}",
        authors=["Researcher et al."],
        year="2023",
        journal="Diabetes Care",
        doi=f"10.2337/{paper_id}",
        section="Results",
        chunk_index=0,
        study_type="Randomized Controlled Trial",
        source="pmc_xml",
        text=text,
        token_count=120,
    )


def test_reversed_metformin_routes_to_true_contradiction_with_positive_weight():
    """
    Task 4: Reproduce reversed-metformin case:
    Same intervention (metformin), same population (T2D), same outcome (cardiovascular mortality).
    Opposing NLI direction.
    MUST route to True Contradiction (SEMANTIC) with weighted_contradiction > 0.
    """
    ref_claim = StructuredClaim(
        chunk_id="query_ref",
        paper_id="query",
        section="Query",
        intervention="metformin",
        population="Type 2 Diabetes",
        outcome="cardiovascular mortality",
        direction="Increase",
        raw_text="Does metformin increase cardiovascular mortality in patients with Type 2 Diabetes?",
    )

    ev_chunk = _make_test_chunk(
        chunk_id="chunk_metformin_01",
        paper_id="paper_metformin_trial",
        text="Metformin significantly reduced cardiovascular mortality in patients with T2D and obesity.",
    )
    ev_claim = StructuredClaim(
        chunk_id="chunk_metformin_01",
        paper_id="paper_metformin_trial",
        section="Results",
        intervention="metformin",
        population="patients with T2D and obesity",
        outcome="cardiovascular mortality",
        direction="Reduction",
        raw_text=ev_chunk.text,
    )
    ev_grade = GRADEAssessment(
        chunk_id="chunk_metformin_01",
        paper_id="paper_metformin_trial",
        tier="HIGH",
        weight=4.0,
        study_design="Randomized Controlled Trial",
        risk_of_bias="Low",
        sample_size_note="n=3000",
        rationale="High certainty landmark trial",
    )
    nli = NLIResult(
        premise_chunk_id="chunk_metformin_01",
        hypothesis_text="Metformin increases cardiovascular mortality in patients with Type 2 Diabetes",
        label="CONTRADICTION",
        confidence=0.98,
        all_scores={"CONTRADICTION": 0.98, "ENTAILMENT": 0.01, "NEUTRAL": 0.01},
        premise_text=ev_chunk.text,
    )

    # 1. Contradiction Analysis
    context_analyses = analyze_contradictions(
        evidence_claims=[ev_claim],
        nli_results=[nli],
        reference_claim=ref_claim,
    )
    assert len(context_analyses) == 1
    analysis = context_analyses[0]
    assert analysis.conflict_type == "SEMANTIC", (
        f"Expected SEMANTIC (True Contradiction), got {analysis.conflict_type}. "
        f"Differences: {[d.description for d in analysis.differences]}"
    )

    # 2. Build Evidence Items and Summarize
    items = build_evidence_items(
        chunks=[ev_chunk],
        claims=[ev_claim],
        nli_results=[nli],
        context_analyses=context_analyses,
        grades=[ev_grade],
    )
    assert len(items) == 1
    assert items[0].relationship == "Contradicts"

    summary = summarize_evidence_relationships(items)
    assert summary.n_contradicting == 1
    assert summary.n_contextual == 0

    # 3. Determine Evidence Status
    status_result = determine_evidence_status(summary, avg_nli_confidence=0.98, avg_retrieval_relevance=0.90)
    assert status_result.n_contradicting == 1
    assert status_result.weighted_contradiction == 4.0
    assert status_result.contextual_weight == 0.0


def test_genuine_contextual_difference_subgroup_divergence_routes_to_contextual():
    """
    Task 5: Reproduce genuine contextual difference:
    Claim A is general/preserved renal function vs Claim B restricted to severe renal failure (eGFR < 30).
    MUST route to Contextual Difference with weighted_contradiction == 0.0 and contextual_weight > 0.0.
    """
    ref_claim = StructuredClaim(
        chunk_id="query_ref",
        paper_id="query",
        section="Query",
        intervention="dapagliflozin",
        population="Type 2 Diabetes with preserved renal function",
        outcome="heart failure hospitalization",
        direction="Reduction",
        raw_text="Dapagliflozin reduces heart failure hospitalization in patients with preserved renal function.",
    )

    ev_chunk = _make_test_chunk(
        chunk_id="chunk_dapa_subgroup",
        paper_id="paper_dapa_esrd",
        text="In patients with severe renal impairment (eGFR < 30), dapagliflozin showed no significant reduction in hospitalization.",
    )
    ev_claim = StructuredClaim(
        chunk_id="chunk_dapa_subgroup",
        paper_id="paper_dapa_esrd",
        section="Results",
        intervention="dapagliflozin",
        population="severe renal impairment (eGFR < 30)",
        outcome="heart failure hospitalization",
        direction="No significant change",
        raw_text=ev_chunk.text,
    )
    ev_grade = GRADEAssessment(
        chunk_id="chunk_dapa_subgroup",
        paper_id="paper_dapa_esrd",
        tier="MODERATE",
        weight=3.0,
        study_design="Randomized Controlled Trial",
        risk_of_bias="Low",
        sample_size_note="n=600",
        rationale="Subgroup trial",
    )
    nli = NLIResult(
        premise_chunk_id="chunk_dapa_subgroup",
        hypothesis_text=ref_claim.raw_text,
        label="CONTRADICTION",
        confidence=0.91,
        all_scores={"CONTRADICTION": 0.91, "ENTAILMENT": 0.03, "NEUTRAL": 0.06},
        premise_text=ev_chunk.text,
    )

    context_analyses = analyze_contradictions(
        evidence_claims=[ev_claim],
        nli_results=[nli],
        reference_claim=ref_claim,
    )
    assert len(context_analyses) == 1
    analysis = context_analyses[0]
    assert analysis.conflict_type == "CONTEXTUAL"
    assert any(d.dimension == "population" for d in analysis.differences)

    items = build_evidence_items(
        chunks=[ev_chunk],
        claims=[ev_claim],
        nli_results=[nli],
        context_analyses=context_analyses,
        grades=[ev_grade],
    )
    assert items[0].relationship == "Contextual Difference"

    summary = summarize_evidence_relationships(items)
    assert summary.n_contradicting == 0
    assert summary.n_contextual == 1

    status_result = determine_evidence_status(summary, avg_nli_confidence=0.91, avg_retrieval_relevance=0.85)
    assert status_result.weighted_contradiction == 0.0
    assert status_result.contextual_weight == 3.0


def test_two_independent_contradicting_studies_reach_refuted_end_to_end():
    """
    Task 6: Construct synthetic case with >=2 independent True Contradiction studies
    satisfying the FR-16.3 replication gate and assert pipeline reaches REFUTED.
    """
    ref_claim = StructuredClaim(
        chunk_id="query_ref",
        paper_id="query",
        section="Query",
        intervention="metformin",
        population="Type 2 Diabetes",
        outcome="cardiovascular mortality",
        direction="Increase",
        raw_text="Does metformin increase cardiovascular mortality in Type 2 Diabetes?",
    )

    chunk1 = _make_test_chunk(
        chunk_id="chunk_trial_a",
        paper_id="PMID_11111111",
        text="Trial A: Metformin significantly decreased cardiovascular mortality in T2D patients.",
    )
    claim1 = StructuredClaim(
        chunk_id="chunk_trial_a",
        paper_id="PMID_11111111",
        section="Results",
        intervention="metformin",
        population="Type 2 Diabetes",
        outcome="cardiovascular mortality",
        direction="Reduction",
        raw_text=chunk1.text,
    )
    grade1 = GRADEAssessment(
        chunk_id="chunk_trial_a",
        paper_id="PMID_11111111",
        tier="HIGH",
        weight=4.0,
        study_design="Randomized Controlled Trial",
        risk_of_bias="Low",
        sample_size_note="n=4000",
        rationale="High certainty RCT",
    )
    nli1 = NLIResult(
        premise_chunk_id="chunk_trial_a",
        hypothesis_text=ref_claim.raw_text,
        label="CONTRADICTION",
        confidence=0.96,
        all_scores={"CONTRADICTION": 0.96},
        premise_text=chunk1.text,
    )

    chunk2 = _make_test_chunk(
        chunk_id="chunk_trial_b",
        paper_id="PMID_22222222",
        text="Trial B: Metformin was associated with significant reduction in cardiovascular death.",
    )
    claim2 = StructuredClaim(
        chunk_id="chunk_trial_b",
        paper_id="PMID_22222222",
        section="Results",
        intervention="metformin",
        population="Type 2 Diabetes",
        outcome="cardiovascular mortality",
        direction="Reduction",
        raw_text=chunk2.text,
    )
    grade2 = GRADEAssessment(
        chunk_id="chunk_trial_b",
        paper_id="PMID_22222222",
        tier="HIGH",
        weight=4.0,
        study_design="Randomized Controlled Trial",
        risk_of_bias="Low",
        sample_size_note="n=5200",
        rationale="High certainty RCT",
    )
    nli2 = NLIResult(
        premise_chunk_id="chunk_trial_b",
        hypothesis_text=ref_claim.raw_text,
        label="CONTRADICTION",
        confidence=0.94,
        all_scores={"CONTRADICTION": 0.94},
        premise_text=chunk2.text,
    )

    context_analyses = analyze_contradictions(
        evidence_claims=[claim1, claim2],
        nli_results=[nli1, nli2],
        reference_claim=ref_claim,
    )
    assert len(context_analyses) == 2
    assert all(ca.conflict_type == "SEMANTIC" for ca in context_analyses)

    items = build_evidence_items(
        chunks=[chunk1, chunk2],
        claims=[claim1, claim2],
        nli_results=[nli1, nli2],
        context_analyses=context_analyses,
        grades=[grade1, grade2],
    )
    assert all(it.relationship == "Contradicts" for it in items)

    summary = summarize_evidence_relationships(items)
    assert summary.n_contradicting == 2
    assert summary.n_supporting == 0

    status_result = determine_evidence_status(summary, avg_nli_confidence=0.95, avg_retrieval_relevance=0.88)
    assert status_result.status == REFUTED
    assert status_result.n_contradicting == 2
    assert status_result.weighted_contradiction == 8.0
    assert "refutes" in status_result.rationale.lower()
