# =============================================================================
# tests/test_dual_mode_qa.py
#
# Tests for Query Intent Classification and Dual-Mode Prompt Generation.
# =============================================================================

import pytest
from src.claims.query_normalizer import classify_query_intent
from src.generation.prompt_builder import build_descriptive_qa_prompt, DESCRIPTIVE_QA_SYSTEM_PROMPT


def test_classify_query_intent_verification():
    verification_queries = [
        "Does metformin reduce cardiovascular risk in patients with type 2 diabetes?",
        "Is empagliflozin superior to glimepiride in reducing HbA1c?",
        "Can SGLT2 inhibitors cause euglycemic diabetic ketoacidosis?",
        "Has semaglutide demonstrated non-inferiority to dulaglutide?",
        "Should sulfonylureas be avoided in elderly patients?",
    ]
    for q in verification_queries:
        assert classify_query_intent(q) == "VERIFICATION", f"Expected VERIFICATION for '{q}'"


def test_classify_query_intent_descriptive():
    descriptive_queries = [
        "What are the common adverse effects of metformin in type 2 diabetes?",
        "How does GLP-1 receptor agonist mechanism of action lower blood glucose?",
        "Why is metformin recommended as first-line pharmacotherapy in T2D?",
        "Describe the cardiovascular and renal outcomes observed with dapagliflozin.",
        "List the major contraindications for sodium-glucose cotransporter-2 inhibitors.",
    ]
    for q in descriptive_queries:
        assert classify_query_intent(q) == "DESCRIPTIVE_QA", f"Expected DESCRIPTIVE_QA for '{q}'"


def test_build_descriptive_qa_prompt():
    evidence_items = [
        {
            "paper_id": "PMID:12345",
            "title": "Adverse effects of metformin therapy",
            "year": 2021,
            "journal": "Diabetes Care",
            "study_type": "Systematic Review",
            "text": "Gastrointestinal disturbances including diarrhea, nausea, and abdominal cramping occur in up to 25% of patients.",
            "grade": {
                "grade_certainty": "HIGH",
                "weight": 4.0,
            },
        }
    ]

    normalized_query = {
        "interventions": ["metformin"],
        "outcomes": ["adverse effects"],
        "population": ["type 2 diabetes"],
    }

    prompt = build_descriptive_qa_prompt(
        question="What are the adverse effects of metformin?",
        evidence_items=evidence_items,
        normalized_query=normalized_query,
    )

    assert "What are the adverse effects of metformin?" in prompt
    assert "Diabetes Care" in prompt
    assert "Adverse effects of metformin therapy" in prompt
    assert "Gastrointestinal disturbances" in prompt
    assert len(DESCRIPTIVE_QA_SYSTEM_PROMPT) > 50
