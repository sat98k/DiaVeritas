# =============================================================================
# tests/test_nli.py
#
# Tests for the NLI classifier.
# These tests verify the interface and basic behavior without requiring
# the full model download (uses mocking for expensive model loads).
#
# Run with: py -m pytest tests/test_nli.py -v
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from src.nli.nli_classifier import (
    NLIResult, classify_nli, summarize_nli_results,
    classify_evidence_against_query,
)
from src.claims.claim_extractor import StructuredClaim


# ---------------------------------------------------------------------------
# NLIResult tests (no model needed)
# ---------------------------------------------------------------------------

class TestNLIResult:
    def make_result(self, label, confidence):
        return NLIResult(
            premise_chunk_id="test_001",
            hypothesis_text="Metformin reduces cardiovascular events.",
            label=label,
            confidence=confidence,
            all_scores={
                "ENTAILMENT": confidence if label == "ENTAILMENT" else 0.1,
                "CONTRADICTION": confidence if label == "CONTRADICTION" else 0.1,
                "NEUTRAL": confidence if label == "NEUTRAL" else 0.1,
            },
        )

    def test_entailment_high_confidence(self):
        r = self.make_result("ENTAILMENT", 0.85)
        assert r.is_entailment is True
        assert r.is_contradiction is False

    def test_entailment_low_confidence(self):
        r = self.make_result("ENTAILMENT", 0.3)
        # Below threshold (0.5 default)
        assert r.is_entailment is False

    def test_contradiction_high_confidence(self):
        r = self.make_result("CONTRADICTION", 0.75)
        assert r.is_contradiction is True

    def test_neutral(self):
        r = self.make_result("NEUTRAL", 0.9)
        assert r.is_neutral is True
        assert r.is_entailment is False
        assert r.is_contradiction is False

    def test_to_dict(self):
        r = self.make_result("ENTAILMENT", 0.8)
        d = r.to_dict()
        assert d["label"] == "ENTAILMENT"
        assert d["confidence"] == 0.8
        assert "all_scores" in d


# ---------------------------------------------------------------------------
# summarize_nli_results tests
# ---------------------------------------------------------------------------

class TestSummarizeNLIResults:
    def make_result(self, label, confidence=0.8):
        return NLIResult(
            premise_chunk_id="test",
            hypothesis_text="test hypothesis",
            label=label,
            confidence=confidence,
            all_scores={"ENTAILMENT": 0.8, "CONTRADICTION": 0.1, "NEUTRAL": 0.1},
        )

    def test_empty_results(self):
        summary = summarize_nli_results([])
        assert summary["total"] == 0
        assert summary["counts"]["ENTAILMENT"] == 0

    def test_all_entailment(self):
        results = [self.make_result("ENTAILMENT") for _ in range(5)]
        summary = summarize_nli_results(results)
        assert summary["counts"]["ENTAILMENT"] == 5
        assert summary["counts"]["CONTRADICTION"] == 0
        assert summary["ratios"]["ENTAILMENT"] == 1.0

    def test_mixed_results(self):
        results = (
            [self.make_result("ENTAILMENT")] * 3 +
            [self.make_result("CONTRADICTION")] * 2 +
            [self.make_result("NEUTRAL")] * 1
        )
        summary = summarize_nli_results(results)
        assert summary["total"] == 6
        assert summary["counts"]["ENTAILMENT"] == 3
        assert summary["counts"]["CONTRADICTION"] == 2
        assert len(summary["entailments"]) == 3
        assert len(summary["contradictions"]) == 2


# ---------------------------------------------------------------------------
# classify_nli with mocked model
# ---------------------------------------------------------------------------

class TestClassifyNLI:
    def _mock_pipeline_output(self, label, score):
        """Simulate HuggingFace pipeline output."""
        labels = ["entailment", "contradiction", "neutral"]
        outputs = []
        for lbl in labels:
            s = score if lbl == label else (1 - score) / 2
            outputs.append({"label": lbl, "score": s})
        return outputs

    @patch("src.nli.nli_classifier._load_nli_model")
    def test_classify_entailment(self, mock_load):
        mock_pipe = MagicMock()
        mock_pipe.return_value = self._mock_pipeline_output("entailment", 0.85)
        mock_load.return_value = mock_pipe

        result = classify_nli(
            premise="Metformin significantly reduced cardiovascular events.",
            hypothesis="Metformin reduces cardiovascular events.",
            premise_chunk_id="chunk_001",
        )

        assert result.label == "ENTAILMENT"
        assert result.confidence > 0.5
        assert result.premise_chunk_id == "chunk_001"

    @patch("src.nli.nli_classifier._load_nli_model")
    def test_classify_contradiction(self, mock_load):
        mock_pipe = MagicMock()
        mock_pipe.return_value = self._mock_pipeline_output("contradiction", 0.80)
        mock_load.return_value = mock_pipe

        result = classify_nli(
            premise="No significant reduction in cardiovascular events was observed.",
            hypothesis="Metformin significantly reduces cardiovascular events.",
        )

        assert result.label == "CONTRADICTION"
        assert result.is_contradiction is True

    @patch("src.nli.nli_classifier._load_nli_model")
    def test_all_scores_present(self, mock_load):
        mock_pipe = MagicMock()
        mock_pipe.return_value = self._mock_pipeline_output("neutral", 0.7)
        mock_load.return_value = mock_pipe

        result = classify_nli("Some biomedical text.", "Some hypothesis.")
        assert "ENTAILMENT" in result.all_scores
        assert "CONTRADICTION" in result.all_scores
        assert "NEUTRAL" in result.all_scores
