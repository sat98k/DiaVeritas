# =============================================================================
# tests/test_claim_grouper.py
#
# Unit tests for Module 12: ClaimGrouper & Deterministic Concept Gating.
# =============================================================================

import pytest
from src.claims.claim_extractor import StructuredClaim
from src.claims.claim_grouper import ClaimGrouper, ClaimGroup


@pytest.fixture
def grouper():
    return ClaimGrouper()


def test_gate_irrelevant_chunks_intervention_mismatch(grouper):
    """
    Chunks that do not mention the query's primary intervention must be gated
    as neutral (off-target) to avoid spurious contradictions.
    """
    query_pico = {
        "interventions": ["metformin"],
        "outcomes": ["cardiovascular events/risk"],
        "population": ["type 2 diabetes"],
    }

    # Off-target chunk: hormone replacement therapy, no metformin
    off_target_claim = StructuredClaim(
        chunk_id="chunk_hrt_01",
        paper_id="paper_hrt",
        section="Results",
        intervention="estrogen",
        population="postmenopausal women",
        outcome="cardiovascular events",
        direction="No significant change",
        raw_text="Estrogen therapy did not reduce cardiovascular events in postmenopausal women.",
    )

    on_target, neutral, groups = grouper.group_and_filter_claims(
        claims=[off_target_claim],
        normalized_query=query_pico,
    )

    assert len(on_target) == 0
    assert len(neutral) == 1
    claim, reason = neutral[0]
    assert claim.chunk_id == "chunk_hrt_01"
    assert "Intervention does not target 'metformin'" in reason


def test_gate_irrelevant_chunks_passes_on_target(grouper):
    """
    Chunks matching the intervention and outcome are marked as on-target,
    allowing downstream NLI to evaluate them.
    """
    query_pico = {
        "interventions": ["metformin"],
        "outcomes": ["cardiovascular events/risk"],
        "population": ["type 2 diabetes"],
    }

    on_target_claim = StructuredClaim(
        chunk_id="chunk_ukpds_01",
        paper_id="paper_ukpds",
        section="Results",
        intervention="metformin",
        population="type 2 diabetes",
        outcome="cardiovascular mortality",
        direction="Reduction",
        raw_text="Metformin reduced cardiovascular mortality by 39% in overweight T2D patients.",
    )

    on_target, neutral, groups = grouper.group_and_filter_claims(
        claims=[on_target_claim],
        normalized_query=query_pico,
    )

    assert len(on_target) == 1
    assert on_target[0].chunk_id == "chunk_ukpds_01"
    assert len(neutral) == 0


def test_group_claims_by_intervention_and_outcome(grouper):
    """
    Test clustering claims by shared intervention and outcome pairs.
    """
    claims = [
        StructuredClaim(
            chunk_id="c1",
            paper_id="p1",
            section="Results",
            intervention="metformin",
            outcome="hba1c",
            raw_text="Metformin reduces HbA1c.",
        ),
        StructuredClaim(
            chunk_id="c2",
            paper_id="p2",
            section="Results",
            intervention="metformin",
            outcome="hba1c",
            raw_text="Metformin significantly lowers HbA1c levels.",
        ),
        StructuredClaim(
            chunk_id="c3",
            paper_id="p3",
            section="Results",
            intervention="empagliflozin",
            outcome="heart failure",
            raw_text="Empagliflozin reduces heart failure hospitalization.",
        ),
    ]

    normalized_query = {
        "interventions": ["metformin"],
        "outcomes": ["glycemic control (hba1c/glucose)"],
    }

    on_target, neutral, groups = grouper.group_and_filter_claims(
        claims=claims,
        normalized_query=normalized_query,
    )

    assert len(groups) == 2
    met_group = next(g for g in groups if "metformin" in g.intervention.lower())
    assert len(met_group.claims) == 2
    assert met_group.is_query_target is True
