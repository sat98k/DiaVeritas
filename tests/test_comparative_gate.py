# =============================================================================
# tests/test_comparative_gate.py
#
# Comprehensive tests for Comparative Claim Verification & Evidence Gate:
#   8A: "Is metformin more effective than sulfonylureas for reducing HbA1c?"
#   8B: "Which is better for lowering HbA1c: metformin or sulfonylureas?"
#   8C: "Does metformin reduce HbA1c?" (non-comparative preservation)
#   8D: "Do SGLT2 inhibitors reduce heart failure hospitalization?" (non-comparative preservation)
#   8E: Reversed comparative claim: "Are sulfonylureas more effective than metformin for reducing HbA1c?"
#   Priority 1: Exercise heuristic extraction & expansion
#   Priority 2: Calibration of study count factor from decisive evidence
# =============================================================================

import pytest
from src.preprocessing.chunker import Chunk
from src.claims.claim_extractor import StructuredClaim, extract_claim_heuristic
from src.claims.query_normalizer import normalize_query, query_to_hypothesis, expand_interventions
from src.claims.comparative_gate import (
    detect_comparative_query,
    evaluate_comparative_chunk,
    ComparativeQueryInfo,
)
from src.evidence.relationship import build_evidence_items, derive_relationship, EvidenceRelationshipSummary
from src.evidence.status import determine_evidence_status, SUPPORTED, REFUTED, INCONCLUSIVE
from src.nli.nli_classifier import NLIResult


def _make_chunk(chunk_id: str, text: str, title: str = "", paper_id: str = "paper_1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        pmc_id="PMC123",
        title=title,
        authors=["Author A"],
        year="2023",
        journal="Diabetes Care",
        doi="10.2337/dc23-0001",
        section="Results",
        chunk_index=0,
        study_type="Randomized Controlled Trial",
        source="pmc_xml",
        text=text,
        token_count=len(text.split()),
        entities={},
    )


def _make_claim(chunk_id: str, intervention: str, outcome: str, direction: str = "Reduction") -> StructuredClaim:
    return StructuredClaim(
        chunk_id=chunk_id,
        paper_id="paper_1",
        section="Results",
        intervention=intervention,
        population="Type 2 Diabetes",
        comparator="Not reported",
        outcome=outcome,
        direction=direction,
        raw_text="",
    )


class TestComparativeQueryDetection:
    """Verify regex and heuristic detection of comparative query structures."""

    def test_8a_directional_metformin_superior(self):
        q = "Is metformin more effective than sulfonylureas for reducing HbA1c?"
        info = detect_comparative_query(q)
        assert info.is_comparative is True
        assert info.comparative_type == "DIRECTIONAL_A_SUPERIOR"
        assert info.intervention_a.lower() == "metformin"
        assert info.intervention_b.lower() == "sulfonylureas"
        assert info.claimed_superior.lower() == "metformin"
        assert any("hba1c" in t for t in info.target_outcome_terms)

    def test_8b_open_choice_which_is_better(self):
        q = "Which is better for lowering HbA1c: metformin or sulfonylureas?"
        info = detect_comparative_query(q)
        assert info.is_comparative is True
        assert info.comparative_type == "OPEN_CHOICE"
        assert info.intervention_a.lower() == "metformin"
        assert info.intervention_b.lower() == "sulfonylureas"
        assert info.claimed_superior is None

    def test_8c_non_comparative_preservation_metformin(self):
        q = "Does metformin reduce HbA1c?"
        info = detect_comparative_query(q)
        assert info.is_comparative is False
        assert info.comparative_type == "NON_COMPARATIVE"

    def test_8d_non_comparative_preservation_sglt2(self):
        q = "Do SGLT2 inhibitors reduce heart failure hospitalization?"
        info = detect_comparative_query(q)
        assert info.is_comparative is False
        assert info.comparative_type == "NON_COMPARATIVE"

    def test_8e_reversed_directional_sulfonylureas_superior(self):
        q = "Are sulfonylureas more effective than metformin for reducing HbA1c?"
        info = detect_comparative_query(q)
        assert info.is_comparative is True
        assert info.comparative_type == "DIRECTIONAL_A_SUPERIOR"
        assert info.intervention_a.lower() == "sulfonylureas"
        assert info.intervention_b.lower() == "metformin"
        assert info.claimed_superior.lower() == "sulfonylureas"


class TestComparativeEvidenceGate:
    """Verify evidence filtering: combination therapy, guidelines, single-agent efficacy, and equivalence."""

    def setup_method(self):
        self.comp_info = detect_comparative_query(
            "Which is better for lowering HbA1c: metformin or sulfonylureas?"
        )

    def test_single_agent_reclassified_to_neutral(self):
        """Single-agent metformin efficacy does NOT prove superiority over sulfonylureas."""
        chunk = _make_chunk(
            "c1",
            "In patients with newly diagnosed type 2 diabetes, metformin monotherapy resulted in a significant 1.2% reduction in HbA1c over 24 weeks.",
            title="Metformin efficacy in type 2 diabetes",
        )
        claim = _make_claim("c1", "metformin", "HbA1c")
        rel, reason = evaluate_comparative_chunk(chunk, claim, self.comp_info, "ENTAILMENT", 0.95)
        assert rel == "Neutral"
        assert "Single-agent" in reason

    def test_combination_therapy_reclassified_to_neutral(self):
        """Gliclazide added to metformin (combination regimen) does NOT prove monotherapy superiority."""
        chunk = _make_chunk(
            "c2",
            "Use of gliclazide as an add-on therapy to metformin in patients with T2DM resulted in better glycemic control than metformin alone.",
            title="Effects of gliclazide add on metformin on glycemic control",
        )
        claim = _make_claim("c2", "metformin", "HbA1c")
        rel, reason = evaluate_comparative_chunk(chunk, claim, self.comp_info, "ENTAILMENT", 0.90)
        assert rel == "Neutral"
        assert "Combination or add-on" in reason

    def test_guideline_first_line_preference_reclassified_to_neutral(self):
        """Guideline recommending metformin as first-line does not establish monotherapy HbA1c superiority."""
        chunk = _make_chunk(
            "c3",
            "Guidelines recommend metformin as first-line therapy compared to sulfonylureas due to combined effects on HbA1c, weight gain, and hypoglycemia.",
            title="Cardiovascular Outcomes Comparison of Oral Antidiabetic Drugs",
        )
        claim = _make_claim("c3", "metformin", "HbA1c")
        rel, reason = evaluate_comparative_chunk(chunk, claim, self.comp_info, "ENTAILMENT", 0.92)
        assert rel == "Neutral"
        assert "first-line" in reason.lower()

    def test_head_to_head_equivalence_refutes_directional_superiority(self):
        """Head-to-head trial showing equivalent HbA1c reduction refutes a directional superiority claim."""
        directional_query = detect_comparative_query(
            "Is metformin more effective than sulfonylureas for reducing HbA1c?"
        )
        chunk = _make_chunk(
            "c4",
            "In a double-blind randomized head-to-head trial, metformin and glimepiride demonstrated similar reductions in HbA1c (-1.1% vs -1.0%, p=0.45) over 52 weeks.",
            title="Head-to-head comparison of metformin and glimepiride in T2D",
        )
        claim = _make_claim("c4", "metformin", "HbA1c")
        rel, reason = evaluate_comparative_chunk(chunk, claim, directional_query, "ENTAILMENT", 0.88)
        assert rel == "Contradicts"
        assert "equivalent glycemic efficacy" in reason

    def test_direct_superiority_supports_claim(self):
        """Genuine head-to-head trial showing greater reduction supports the superiority claim."""
        directional_query = detect_comparative_query(
            "Is metformin more effective than sulfonylureas for reducing HbA1c?"
        )
        chunk = _make_chunk(
            "c5",
            "In head-to-head comparison, metformin produced greater reduction in HbA1c compared to sulfonylureas (-1.5% vs -0.8%, p<0.01).",
            title="Superior glycemic control with metformin vs sulfonylureas",
        )
        claim = _make_claim("c5", "metformin", "HbA1c")
        rel, reason = evaluate_comparative_chunk(chunk, claim, directional_query, "ENTAILMENT", 0.94)
        assert rel == "Supports"


class TestEndToEndComparativeGate:
    """Verify build_evidence_items integration and final status determination."""

    def test_metformin_vs_su_inconclusive_verdict(self):
        """When evidence contains only combo, guideline, and single-agent chunks, verdict must be INCONCLUSIVE."""
        q = "Which is better for lowering HbA1c in Type 2 Diabetes: metformin or sulfonylureas?"
        norm = normalize_query(q)

        c1 = _make_chunk("c1", "Metformin reduces HbA1c significantly in T2D patients.")
        c2 = _make_chunk("c2", "Sulfonylureas improve glycemic control by stimulating insulin secretion.")
        c3 = _make_chunk("c3", "Metformin-sulfonylurea combination therapy helps patients reach HbA1c target compared to insulin.")
        c4 = _make_chunk("c4", "Guidelines recommend metformin as first-line therapy over sulfonylureas due to weight neutrality.")

        chunks = [c1, c2, c3, c4]
        claims = [_make_claim(c.chunk_id, "metformin", "HbA1c") for c in chunks]
        # Raw NLI without gate might predict ENTAILMENT
        nli_results = [
            NLIResult(c.chunk_id, norm["hypothesis_text"], "ENTAILMENT", 0.90, {})
            for c in chunks
        ]

        items = build_evidence_items(
            chunks=chunks,
            claims=claims,
            nli_results=nli_results,
            context_analyses=[],
            query_context=norm,
        )

        # All 4 items should be gated to Neutral
        for it in items:
            assert it.relationship == "Neutral", f"Item {it.chunk.chunk_id} should be Neutral, got {it.relationship}"

        summary = EvidenceRelationshipSummary(total=len(items))
        summary.neutral = items
        res = determine_evidence_status(summary)

        assert res.status == INCONCLUSIVE
        assert res.n_supporting == 0
        assert res.confidence_score <= 0.65  # Calibrated, not inflated to 0.92


class TestPriority1ExerciseExtraction:
    """Verify Priority 1 fix: Exercise intervention/outcome heuristic extraction & expansion."""

    def test_exercise_heuristic_extraction_when_drugs_empty(self):
        chunk = _make_chunk(
            "ex1",
            "Resistance training improved muscle mass and muscle strength in older adults with type 2 diabetes.",
        )
        claim = extract_claim_heuristic(chunk)
        assert claim.intervention == "resistance training"
        assert claim.outcome == "muscle mass"

    def test_exercise_expansion_includes_modalities(self):
        exp = expand_interventions(["exercise/physical activity"])
        assert "resistance training" in exp
        assert "aerobic exercise" in exp
        assert "walking" in exp


class TestPriority2ConfidenceCalibration:
    """Verify Priority 2 fix: Confidence study count calculated from decisive items."""

    def test_zero_decisive_studies_has_low_study_factor(self):
        from src.evidence.status import _compute_confidence

        summary = EvidenceRelationshipSummary(total=8)
        # 8 neutral chunks from 8 different papers
        summary.neutral = [
            _make_chunk(f"c{i}", "Neutral text", paper_id=f"paper_{i}")
            for i in range(8)
        ]
        conf_score, breakdown = _compute_confidence(
            status=INCONCLUSIVE,
            entailment_ratio=0.0,
            contradiction_ratio=0.0,
            n_total=8,
            summary=summary,
            avg_nli_confidence=0.7,
        )
        # study_count factor should be 0.25 (0 decisive studies), NOT inflated to 1.0
        assert breakdown["independent_study_count"] == 0.25
        assert conf_score < 0.60
