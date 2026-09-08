# =============================================================================
# tests/test_query_normalizer.py
#
# Unit tests for query normalization, concept retention, and hypothesis creation.
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.claims.query_normalizer import normalize_query, query_to_hypothesis


def test_exercise_query_preserves_terms():
    """Verify that exercise queries retain lifestyle terms in normalized_text and interventions."""
    q = "Does diabetes improve with exercise (like walking)?"
    result = normalize_query(q)

    assert "exercise/physical activity" in result["interventions"]
    assert "clinical improvement" in result["outcomes"]
    assert "walking" in result["normalized_text"].lower()
    assert "exercise" in result["normalized_text"].lower()
    assert result["dense_query"] == q


def test_lifestyle_keywords_detected():
    """Verify diet, yoga, resistance training are properly detected as interventions."""
    res_diet = normalize_query("Does intermittent fasting lower blood glucose in T2D?")
    assert "fasting/intermittent fasting" in res_diet["interventions"]

    res_yoga = normalize_query("Is yoga effective for glycemic control in diabetes?")
    assert "yoga" in res_yoga["interventions"]

    res_aerobic = normalize_query("Long-term effects of aerobic exercise on HbA1c")
    assert "aerobic exercise" in res_aerobic["interventions"]


def test_dense_query_is_original():
    """Ensure dense_query carries the complete natural language question for vector embeddings."""
    q = "Can lifestyle interventions reverse prediabetes?"
    res = normalize_query(q)
    assert res["dense_query"] == q


def test_drug_query_still_works():
    """Verify pharmacological queries continue to extract drug entities and disease concepts."""
    q = "Does metformin reduce cardiovascular risk in patients with Type 2 Diabetes?"
    res = normalize_query(q)
    assert "metformin" in res["interventions"]
    assert any("diabetes" in d.lower() for d in res["disease"])
    assert "cardiovascular events/risk" in res["outcomes"]


def test_outcome_improve_detected():
    """Verify that 'improve' and 'better' map to clinical improvement."""
    res = normalize_query("Do SGLT2 inhibitors improve heart failure outcomes?")
    assert "clinical improvement" in res["outcomes"]
    assert "heart failure" in res["outcomes"]
