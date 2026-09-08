# =============================================================================
# tests/test_evidence_status.py
#
# Tests for evidence relationship derivation and status determination.
# Run with: py -m pytest tests/test_evidence_status.py -v
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.evidence.status import (
    determine_evidence_status, SUPPORTED, REFUTED, INCONCLUSIVE,
)
from src.evidence.relationship import (
    EvidenceRelationshipSummary, derive_relationship,
)
from src.nli.nli_classifier import NLIResult
from src.context.contradiction_analyzer import ContextComparisonResult


# ---------------------------------------------------------------------------
# Evidence relationship tests
# ---------------------------------------------------------------------------

class TestDeriveRelationship:
    def make_nli(self, label, confidence):
        return NLIResult(
            premise_chunk_id="test",
            hypothesis_text="test",
            label=label,
            confidence=confidence,
            all_scores={},
        )

    def test_high_confidence_entailment_supports(self):
        nli = self.make_nli("ENTAILMENT", 0.85)
        assert derive_relationship(nli) == "Supports"

    def test_low_confidence_entailment_neutral(self):
        nli = self.make_nli("ENTAILMENT", 0.3)  # Below 0.5 threshold
        assert derive_relationship(nli) == "Neutral"

    def test_high_confidence_contradiction(self):
        nli = self.make_nli("CONTRADICTION", 0.75)
        assert derive_relationship(nli) == "Contradicts"

    def test_contradiction_with_contextual_analysis(self):
        nli = self.make_nli("CONTRADICTION", 0.75)
        ctx = ContextComparisonResult(
            claim_a_id="a",
            claim_b_id="b",
            nli_label="CONTRADICTION",
            nli_confidence=0.75,
            conflict_type="CONTEXTUAL",
        )
        assert derive_relationship(nli, ctx) == "Contextual Difference"

    def test_neutral_label(self):
        nli = self.make_nli("NEUTRAL", 0.9)
        assert derive_relationship(nli) == "Neutral"


# ---------------------------------------------------------------------------
# Evidence status tests
# ---------------------------------------------------------------------------

def make_summary(n_sup=0, n_con=0, n_ctx=0, n_neu=0):
    """Create a minimal EvidenceRelationshipSummary for testing."""
    # Create dummy items (we only need the counts for status testing)
    summary = EvidenceRelationshipSummary(
        total=n_sup + n_con + n_ctx + n_neu,
    )
    # We can't easily mock the lists without real EvidenceItems,
    # so we test via the counts indirectly through a mock summary.
    # For simplicity, create a custom summary with count properties mocked.
    summary._sup = [None] * n_sup
    summary._con = [None] * n_con
    summary._ctx = [None] * n_ctx
    summary._neu = [None] * n_neu
    summary.supporting = summary._sup
    summary.contradicting = summary._con
    summary.contextual = summary._ctx
    summary.neutral = summary._neu
    return summary


class TestDetermineEvidenceStatus:
    def test_supported_clear_majority(self):
        """8 supporting, 0 contradicting → SUPPORTED"""
        summary = make_summary(n_sup=8, n_con=0, n_ctx=0, n_neu=2)
        result = determine_evidence_status(summary, avg_nli_confidence=0.8)
        assert result.status == SUPPORTED

    def test_refuted_clear_majority(self):
        """0 supporting, 8 contradicting → REFUTED"""
        summary = make_summary(n_sup=0, n_con=8, n_ctx=0, n_neu=2)
        result = determine_evidence_status(summary, avg_nli_confidence=0.8)
        assert result.status == REFUTED

    def test_supported_with_neutral_majority(self):
        """3 supporting, 0 contradicting, 7 neutral → SUPPORTED (decisive consensus)"""
        summary = make_summary(n_sup=3, n_con=0, n_ctx=0, n_neu=7)
        result = determine_evidence_status(summary, avg_nli_confidence=0.85)
        assert result.status == SUPPORTED
        assert result.confidence_label in ("High", "Moderate")

    def test_insufficient_decisive_stays_inconclusive(self):
        """1 supporting, 0 contradicting, 9 neutral → INCONCLUSIVE (insufficient evidence)"""
        summary = make_summary(n_sup=1, n_con=0, n_ctx=0, n_neu=9)
        result = determine_evidence_status(summary)
        assert result.status == INCONCLUSIVE

    def test_refuted_with_neutral_majority(self):
        """0 supporting, 3 contradicting, 7 neutral → REFUTED (decisive consensus)"""
        summary = make_summary(n_sup=0, n_con=3, n_ctx=0, n_neu=7)
        result = determine_evidence_status(summary, avg_nli_confidence=0.85)
        assert result.status == REFUTED

    def test_inconclusive_mixed(self):
        """4 supporting, 4 contradicting → INCONCLUSIVE"""
        summary = make_summary(n_sup=4, n_con=4, n_ctx=0, n_neu=2)
        result = determine_evidence_status(summary)
        assert result.status == INCONCLUSIVE

    def test_inconclusive_no_evidence(self):
        """No evidence → INCONCLUSIVE"""
        summary = make_summary()
        result = determine_evidence_status(summary)
        assert result.status == INCONCLUSIVE
        assert result.confidence_score == 0.0

    def test_inconclusive_is_valid_outcome(self):
        """INCONCLUSIVE must be returned and must not be treated as error."""
        summary = make_summary(n_sup=2, n_con=3, n_ctx=2, n_neu=3)
        result = determine_evidence_status(summary)
        assert result.status == INCONCLUSIVE
        assert "INCONCLUSIVE" in result.rationale or "mixed" in result.rationale.lower()

    def test_confidence_label_ranges(self):
        from src.evidence.status import _confidence_label
        assert _confidence_label(0.8) == "High"
        assert _confidence_label(0.5) == "Moderate"
        assert _confidence_label(0.2) == "Low"

    def test_to_dict_completeness(self):
        summary = make_summary(n_sup=3, n_con=1, n_ctx=1, n_neu=2)
        result = determine_evidence_status(summary)
        d = result.to_dict()
        required_keys = [
            "status", "confidence_score", "confidence_label",
            "entailment_ratio", "contradiction_ratio", "n_total",
            "n_supporting", "n_contradicting", "rationale",
        ]
        for key in required_keys:
            assert key in d, f"Missing key in status dict: {key}"
