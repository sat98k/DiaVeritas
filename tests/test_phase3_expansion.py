# =============================================================================
# tests/test_phase3_expansion.py
#
# Tests for Phase 3: Query-Time Concept Expansion and Concept Gating
# =============================================================================

import pytest
from src.claims.query_normalizer import normalize_query, expand_interventions, expand_outcomes
from src.claims.claim_grouper import ClaimGrouper
from src.claims.claim_extractor import StructuredClaim


def test_sglt2_class_expansion():
    """Generic drug class query expands to member drugs."""
    q = "Do SGLT2 inhibitors reduce the risk of hospitalization for heart failure in patients with Type 2 Diabetes?"
    norm = normalize_query(q)

    assert "expanded_interventions" in norm
    assert "expanded_outcomes" in norm

    exp_int = norm["expanded_interventions"]
    assert "empagliflozin" in exp_int
    assert "dapagliflozin" in exp_int
    assert "canagliflozin" in exp_int
    assert "sglt2" in exp_int

    exp_out = norm["expanded_outcomes"]
    assert "heart failure" in exp_out
    assert any("hospitalization" in o for o in exp_out)


def test_specific_member_does_not_cross_contaminate():
    """A specific member drug expands to class and brands, but NOT sister drugs."""
    exp = expand_interventions(["empagliflozin"], "Does empagliflozin reduce cardiovascular risk?")
    assert "empagliflozin" in exp
    assert "jardiance" in exp
    assert any("sglt" in x for x in exp)
    assert "dapagliflozin" not in exp
    assert "canagliflozin" not in exp


def test_claim_grouper_admits_member_trials_for_class_query():
    """
    ClaimGrouper must accept empagliflozin and dapagliflozin landmark trial claims
    as on-target for an SGLT2 class query.
    """
    q = "Do SGLT2 inhibitors reduce the risk of hospitalization for heart failure in patients with Type 2 Diabetes?"
    norm = normalize_query(q)

    empa_claim = StructuredClaim(
        chunk_id="empa_01",
        paper_id="empa_reg_outcome",
        section="Results",
        intervention="empagliflozin",
        population="type 2 diabetes and high cardiovascular risk",
        outcome="hospitalization for heart failure",
        direction="Reduction",
        raw_text="In the EMPA-REG OUTCOME trial, empagliflozin significantly reduced hospitalization for heart failure (HR 0.65, 95% CI 0.50-0.85).",
    )

    dapa_claim = StructuredClaim(
        chunk_id="dapa_01",
        paper_id="dapa_hf",
        section="Results",
        intervention="dapagliflozin",
        population="heart failure with reduced ejection fraction",
        outcome="worsening heart failure or cardiovascular death",
        direction="Reduction",
        raw_text="Dapagliflozin reduced the risk of worsening heart failure or cardiovascular death in patients with heart failure.",
    )

    unrelated_claim = StructuredClaim(
        chunk_id="hrt_01",
        paper_id="paper_hrt",
        section="Results",
        intervention="estrogen",
        population="postmenopausal women",
        outcome="bone mineral density",
        direction="Increase",
        raw_text="Estrogen therapy increased bone mineral density in postmenopausal women.",
    )

    grouper = ClaimGrouper()
    on_target, neutral, groups = grouper.group_and_filter_claims(
        claims=[empa_claim, dapa_claim, unrelated_claim],
        normalized_query=norm,
    )

    on_target_ids = [c.chunk_id for c in on_target]
    assert "empa_01" in on_target_ids, "Empagliflozin trial claim should be on-target for SGLT2 query"
    assert "dapa_01" in on_target_ids, "Dapagliflozin trial claim should be on-target for SGLT2 query"

    neutral_ids = [c.chunk_id for c, _ in neutral]
    assert "hrt_01" in neutral_ids, "Estrogen claim should be gated to neutral"
