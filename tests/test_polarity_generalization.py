# =============================================================================
# tests/test_polarity_generalization.py
#
# Unit tests for General Polarity Classification & 10-Point Generalization Matrix.
# Verifies that:
#   - Matching intervention + outcome + compatible population + opposite direction -> Contradicts
#   - Matching intervention + outcome + compatible population + matching direction -> Supports
#   - Population divergence (e.g. pediatric vs adult) -> Contextual Difference or Neutral
#   - Intervention mismatch -> Neutral / Off-target
# =============================================================================

import pytest
from src.preprocessing.chunker import Chunk
from src.claims.claim_extractor import extract_claims, StructuredClaim
from src.claims.query_normalizer import normalize_query, query_to_hypothesis, extract_query_direction
from src.claims.claim_grouper import ClaimGrouper
from src.nli.nli_classifier import classify_evidence_against_query, NLIResult
from src.context.contradiction_analyzer import (
    analyze_contradictions,
    _interventions_match,
    _outcomes_match,
    _populations_compatible,
    _is_opposite_direction,
    _is_matching_direction,
    _normalize_direction,
)
from src.evidence.relationship import derive_relationship, build_evidence_items


def make_test_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id="paper_gen",
        pmc_id="PMC99999",
        title="Biomedical Study",
        authors=["Researcher A"],
        year="2023",
        journal="Diabetes Care",
        doi="10.1234/dc.2023",
        section="Results",
        chunk_index=0,
        study_type="RCT",
        source="pmc_xml",
        text=text,
        token_count=len(text.split()),
        entities={},
    )


def evaluate_polarity_pipeline(evidence_text: str, query_text: str, chunk_id: str = "chunk_1"):
    chunk = make_test_chunk(chunk_id, evidence_text)
    claims = extract_claims([chunk])
    norm_query = normalize_query(query_text)

    grouper = ClaimGrouper()
    on_target, neutral_gated, groups = grouper.group_and_filter_claims(claims, norm_query)

    hyp_text = query_to_hypothesis(query_text)

    if on_target:
        nli_results = classify_evidence_against_query(hyp_text, on_target)
        nli_res = nli_results[0]
    else:
        reason = neutral_gated[0][1] if neutral_gated else "Off-target"
        nli_res = NLIResult(
            premise_chunk_id=chunk_id,
            hypothesis_text=hyp_text,
            label="NEUTRAL",
            confidence=0.85,
            all_scores={"ENTAILMENT": 0.05, "CONTRADICTION": 0.10, "NEUTRAL": 0.85},
            premise_text=f"Off-target concept: {reason}",
        )

    target_ints = norm_query.get("interventions", [])
    target_outs = norm_query.get("outcomes", [])
    target_pop = norm_query.get("population", [])
    if not target_pop and norm_query.get("disease"):
        target_pop = norm_query.get("disease")

    ref_claim = StructuredClaim(
        chunk_id="query_ref",
        paper_id="query",
        section="Query",
        intervention=target_ints[0] if target_ints else "Not reported",
        outcome=target_outs[0] if target_outs else "Not reported",
        population=target_pop[0] if target_pop else "Type 2 Diabetes",
        direction=extract_query_direction(query_text),
        raw_text=query_text,
    )

    context_analyses = analyze_contradictions(claims, [nli_res], reference_claim=ref_claim)

    items = build_evidence_items(
        chunks=[chunk],
        claims=claims,
        nli_results=[nli_res],
        context_analyses=context_analyses,
        query_context=norm_query,
    )
    return items[0], claims[0], ref_claim, nli_res, context_analyses


class TestGenericMatchingHelpers:
    """Test generic PICO matching helper functions."""

    def test_interventions_match_exact_and_stems(self):
        assert _interventions_match("metformin", "metformin")
        assert _interventions_match("metformin therapy", "metformin monotherapy")

    def test_interventions_match_drug_classes(self):
        assert _interventions_match("sglt2 inhibitor", "empagliflozin")
        assert _interventions_match("sglt2 inhibitors", "dapagliflozin")
        assert _interventions_match("glp-1 receptor agonist", "semaglutide")

    def test_interventions_mismatch_different_drugs(self):
        assert not _interventions_match("empagliflozin", "metformin")
        assert not _interventions_match("sulfonylureas", "metformin")

    def test_interventions_match_lifestyle(self):
        assert _interventions_match("exercise", "aerobic exercise")
        assert _interventions_match("exercise", "resistance training")
        assert _interventions_match("resistance training", "strength training")
        assert not _interventions_match("aerobic exercise", "resistance training")

    def test_outcomes_match_clusters(self):
        assert _outcomes_match("cardiovascular mortality", "mortality")
        assert _outcomes_match("hospitalization for heart failure", "heart failure")
        assert _outcomes_match("hba1c", "glycemic control")
        assert _outcomes_match("skeletal muscle mass and strength", "muscle mass")

    def test_populations_compatible(self):
        compat, _ = _populations_compatible("patients with type 2 diabetes", "adults with T2D")
        assert compat
        compat, desc = _populations_compatible("pediatric patients", "adults with type 2 diabetes")
        assert not compat
        assert "pediatric" in desc.lower()

    def test_direction_helpers(self):
        assert _is_opposite_direction("REDUCTION", "INCREASE")
        assert _is_opposite_direction("INCREASE", "REDUCTION")
        assert not _is_opposite_direction("REDUCTION", "REDUCTION")
        assert _is_matching_direction("REDUCTION", "REDUCTION")
        assert _is_matching_direction("INCREASE", "INCREASE")


class TestGeneralizationMatrix:
    """The 10-point Generalization Test Matrix requested by user."""

    def test_matrix_1_metformin_cv_mortality_matching(self):
        ev = "Metformin reduces cardiovascular mortality in T2D."
        q = "Does metformin reduce cardiovascular mortality in T2D?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_1")
        assert item.relationship == "Supports"

    def test_matrix_2_metformin_cv_mortality_opposite(self):
        ev = "Metformin reduces cardiovascular mortality in T2D."
        q = "Does metformin increase cardiovascular mortality in T2D?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_2")
        assert item.relationship == "Contradicts"

    def test_matrix_3_sglt2_heart_failure_opposite(self):
        ev = "Empagliflozin significantly reduced the risk of hospitalization for heart failure compared with placebo in patients with type 2 diabetes."
        q = "Do SGLT2 inhibitors increase hospitalization for heart failure in Type 2 Diabetes?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_3")
        assert item.relationship == "Contradicts"

    def test_matrix_4_sglt2_heart_failure_matching(self):
        ev = "Empagliflozin significantly reduced the risk of hospitalization for heart failure compared with placebo in patients with type 2 diabetes."
        q = "Do SGLT2 inhibitors reduce hospitalization for heart failure in Type 2 Diabetes?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_4")
        assert item.relationship == "Supports"

    def test_matrix_5_aerobic_exercise_hba1c_opposite(self):
        ev = "Aerobic exercise training leads to a clinically significant reduction in HbA1c levels in adults with type 2 diabetes."
        q = "Does aerobic exercise increase HbA1c in Type 2 Diabetes?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_5")
        assert item.relationship == "Contradicts"

    def test_matrix_6_aerobic_exercise_hba1c_matching(self):
        ev = "Aerobic exercise training leads to a clinically significant reduction in HbA1c levels in adults with type 2 diabetes."
        q = "Does aerobic exercise reduce HbA1c in Type 2 Diabetes?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_6")
        assert item.relationship == "Supports"

    def test_matrix_7_resistance_training_muscle_mass_opposite(self):
        ev = "Resistance training increases skeletal muscle mass and strength in older adults with type 2 diabetes."
        q = "Does resistance training decrease muscle mass in Type 2 Diabetes?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_7")
        assert item.relationship == "Contradicts"

    def test_matrix_8_resistance_training_muscle_mass_matching(self):
        ev = "Resistance training increases skeletal muscle mass and strength in older adults with type 2 diabetes."
        q = "Does resistance training increase muscle mass in Type 2 Diabetes?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_8")
        assert item.relationship == "Supports"

    def test_matrix_9_population_divergence(self):
        ev = "Intervention X showed no benefit in pediatric patients."
        q = "Does intervention X reduce mortality in adults with Type 2 Diabetes?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_9")
        assert item.relationship in ("Contextual Difference", "Neutral")
        assert item.relationship != "Contradicts"

    def test_matrix_10_intervention_mismatch(self):
        ev = "Empagliflozin reduces cardiovascular mortality."
        q = "Does metformin reduce cardiovascular mortality?"
        item, _, _, _, _ = evaluate_polarity_pipeline(ev, q, "matrix_10")
        assert item.relationship in ("Neutral", "Contextual Difference")
        assert item.relationship != "Supports"
