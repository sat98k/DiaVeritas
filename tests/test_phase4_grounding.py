# =============================================================================
# tests/test_phase4_grounding.py
#
# Tests for Phase 4: Strict Citation Grounding Enforcement (NFR-5.2)
# =============================================================================

import pytest
from src.generation.synthesizer import enforce_strict_grounding
from src.evidence.relationship import EvidenceItem
from src.preprocessing.chunker import Chunk
from src.claims.claim_extractor import StructuredClaim
from src.nli.nli_classifier import NLIResult


def _make_mock_evidence(texts):
    items = []
    for i, text in enumerate(texts):
        c = Chunk(
            chunk_id=f"c_{i}",
            paper_id=f"p_{i}",
            pmc_id="PMC12345",
            title="SGLT2 Trial",
            authors=["Zinman B"],
            year="2015",
            journal="N Engl J Med",
            doi="10.1056/NEJMoa1504720",
            section="Results",
            chunk_index=i,
            study_type="RCT",
            source="pmc_xml",
            text=text,
            token_count=len(text.split()),
        )
        claim = StructuredClaim(
            chunk_id=f"c_{i}",
            paper_id=f"p_{i}",
            section="Results",
            raw_text=text,
        )
        nli = NLIResult(
            premise_chunk_id=f"c_{i}",
            hypothesis_text="test",
            label="ENTAILMENT",
            confidence=0.9,
            all_scores={},
        )
        items.append(EvidenceItem(chunk=c, claim=claim, nli_result=nli, relationship="Supports"))
    return items


def test_grounded_sentence_with_citation_preserved():
    evidence = _make_mock_evidence([
        "Empagliflozin significantly reduced hospitalization for heart failure in patients with type 2 diabetes."
    ])

    draft = (
        "Empagliflozin significantly reduced hospitalization for heart failure [Zinman 2015, N Engl J Med]. "
        "Overall findings demonstrate clinical efficacy in patients with type 2 diabetes [Zinman 2015, N Engl J Med]."
    )

    clean_answer, stripped = enforce_strict_grounding(draft, evidence)

    assert "[Zinman 2015, N Engl J Med]" in clean_answer
    assert len(stripped) == 0, "No grounded sentence should be stripped"


def test_ungrounded_speculative_sentence_stripped():
    evidence = _make_mock_evidence([
        "Empagliflozin significantly reduced hospitalization for heart failure in patients with type 2 diabetes."
    ])

    draft = (
        "Empagliflozin significantly reduced hospitalization for heart failure [Zinman 2015, N Engl J Med]. "
        "Patients taking this medication should also consume large quantities of green tea daily to boost antioxidant absorption."
    )

    clean_answer, stripped = enforce_strict_grounding(draft, evidence)

    assert "[Zinman 2015, N Engl J Med]" in clean_answer
    assert "green tea" not in clean_answer, "Ungrounded sentence must be programmatically stripped"
    assert len(stripped) == 1
    assert "green tea" in stripped[0]


def test_lexical_grounding_preserved_even_without_inline_brackets():
    evidence = _make_mock_evidence([
        "Empagliflozin significantly reduced hospitalization for heart failure in patients with type 2 diabetes."
    ])

    draft = (
        "In patients with type 2 diabetes, empagliflozin significantly reduced hospitalization for heart failure."
    )

    clean_answer, stripped = enforce_strict_grounding(draft, evidence)

    assert "empagliflozin" in clean_answer
    assert len(stripped) == 0


def test_citation_attached_to_unrelated_passage_is_stripped():
    """
    Phase 4 Acceptance Test:
    A sentence that attaches a valid citation marker to a completely unrelated/unsupported
    claim MUST be detected as misattributed and programmatically stripped, while the
    legitimately supported sentence is preserved.
    """
    evidence = _make_mock_evidence([
        "Empagliflozin significantly reduced hospitalization for heart failure in patients with type 2 diabetes."
    ])

    draft = (
        "Empagliflozin significantly reduced hospitalization for heart failure [Zinman 2015, N Engl J Med]. "
        "Metformin completely cures neurological tremors and Parkinsonism within two weeks [Zinman 2015, N Engl J Med]."
    )

    clean_answer, stripped = enforce_strict_grounding(draft, evidence)

    assert "[Zinman 2015, N Engl J Med]" in clean_answer
    assert "Empagliflozin significantly reduced hospitalization" in clean_answer
    assert "neurological tremors" not in clean_answer, "Misattributed sentence must be programmatically stripped"
    assert len(stripped) == 1
    assert "neurological tremors" in stripped[0]
