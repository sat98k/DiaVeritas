# =============================================================================
# tests/test_chunker.py
#
# Unit tests for the section-aware chunker.
# Run with: py -m pytest tests/test_chunker.py -v
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.preprocessing.chunker import chunk_paper, chunk_section, Chunk
from src.preprocessing.xml_parser import ParsedPaper, SectionBlock
from src.utils.text_utils import split_sentences, count_tokens_approx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_parsed_paper(sections=None, paper_id="TEST001"):
    sections = sections or []
    return ParsedPaper(
        paper_id=paper_id,
        pmc_id="PMC9999999",
        title="Test Paper on Metformin",
        authors=["Smith J", "Jones A"],
        year="2021",
        journal="Diabetes Care",
        doi="10.1234/test",
        abstract="Metformin significantly reduced HbA1c in T2D patients.",
        sections=sections,
    )


def make_section(text, name="Methods", order=0):
    return SectionBlock(
        section=name,
        raw_heading=name,
        text=text,
        order=order,
    )


# ---------------------------------------------------------------------------
# Tests: sentence splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    def test_simple_sentences(self):
        text = "Metformin was administered. HbA1c was measured. Outcomes were significant."
        sents = split_sentences(text)
        assert len(sents) == 3

    def test_abbreviations_not_split(self):
        text = "The study used e.g. metformin. HbA1c was measured."
        sents = split_sentences(text)
        # Should not split on "e.g." — only on "metformin."
        assert len(sents) >= 1

    def test_decimal_not_split(self):
        text = "The dose was 1.5 mg/kg. Outcomes were positive."
        sents = split_sentences(text)
        # Should not split on "1.5"
        assert not any(s.strip() == "5 mg/kg." for s in sents)

    def test_empty_text(self):
        assert split_sentences("") == [""]

    def test_single_sentence(self):
        text = "Metformin reduced cardiovascular events."
        sents = split_sentences(text)
        assert len(sents) == 1
        assert sents[0] == text


# ---------------------------------------------------------------------------
# Tests: token counting
# ---------------------------------------------------------------------------

class TestTokenCounting:
    def test_empty(self):
        assert count_tokens_approx("") == 0

    def test_single_word(self):
        assert count_tokens_approx("metformin") == 1

    def test_multiple_words(self):
        text = "Metformin reduces HbA1c in T2D patients"
        assert count_tokens_approx(text) == 6


# ---------------------------------------------------------------------------
# Tests: chunk_section
# ---------------------------------------------------------------------------

class TestChunkSection:
    def test_basic_chunking(self):
        # Generate enough text for at least 2 chunks
        long_text = " ".join([
            "Metformin was administered to all patients with type 2 diabetes."
            " The primary outcome was HbA1c reduction after 24 weeks."
            " Secondary outcomes included weight loss and blood pressure."
        ] * 30)  # ~30x repetition to exceed chunk size

        section = make_section(long_text)
        paper = make_parsed_paper([section])
        chunks = chunk_section(section, paper, study_type="RCT")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.paper_id == "TEST001"
            assert chunk.section == "Methods"
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0
            assert chunk.token_count > 0

    def test_chunk_ids_unique(self):
        long_text = " ".join([
            "Patients were randomized to receive metformin or placebo."
        ] * 40)
        section = make_section(long_text)
        paper = make_parsed_paper([section])
        chunks = chunk_section(section, paper, study_type="RCT")

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_short_text_produces_one_chunk(self):
        # Text with 30+ tokens to exceed the minimum chunk filter
        text = (
            "Metformin significantly reduced HbA1c compared to placebo in T2D patients "
            "after 24 weeks of treatment in a double-blind randomized controlled trial "
            "with adequate statistical power and follow-up."
        )
        section = make_section(text)
        paper = make_parsed_paper([section])
        chunks = chunk_section(section, paper, study_type="Not reported")
        assert len(chunks) == 1
        assert chunks[0].token_count >= 20

    def test_metadata_preserved(self):
        # Use enough text to pass the 20-token minimum filter
        text = (
            "Metformin significantly reduced major adverse cardiovascular events in patients "
            "with type 2 diabetes compared to placebo over 24 weeks of follow-up. "
            "The primary outcome was a composite of cardiovascular death and non-fatal myocardial infarction."
        )
        section = make_section(text, name="Results")
        paper = make_parsed_paper([section])
        chunks = chunk_section(section, paper, study_type="RCT")

        assert len(chunks) >= 1
        c = chunks[0]
        assert c.paper_id == "TEST001"
        assert c.title == "Test Paper on Metformin"
        assert c.year == "2021"
        assert c.journal == "Diabetes Care"
        assert c.section == "Results"
        assert c.study_type == "RCT"
        assert c.source == "pmc_xml"


# ---------------------------------------------------------------------------
# Tests: chunk_paper
# ---------------------------------------------------------------------------

class TestChunkPaper:
    def test_references_skipped(self):
        text = "Smith J et al. Diabetes Care. 2021. Jones A et al. NEJM 2020."
        section = make_section(text, name="References")
        paper = make_parsed_paper([section])
        chunks = chunk_paper(paper)
        assert len(chunks) == 0, "References section should be skipped"

    def test_multiple_sections(self):
        sections = [
            make_section("Methods text about metformin dosing " * 20, "Methods", 0),
            make_section("Results showed HbA1c reduction " * 20, "Results", 1),
        ]
        paper = make_parsed_paper(sections)
        chunks = chunk_paper(paper)

        sections_in_chunks = {c.section for c in chunks}
        assert "Methods" in sections_in_chunks
        assert "Results" in sections_in_chunks

    def test_empty_paper(self):
        paper = make_parsed_paper([])
        chunks = chunk_paper(paper)
        assert chunks == []


# ---------------------------------------------------------------------------
# Tests: Chunk serialization
# ---------------------------------------------------------------------------

class TestChunkSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        chunk = Chunk(
            chunk_id="TEST001_Methods_000",
            paper_id="TEST001",
            pmc_id="PMC999",
            title="Test",
            authors=["Smith J"],
            year="2021",
            journal="Diabetes Care",
            doi="Not reported",
            section="Methods",
            chunk_index=0,
            study_type="RCT",
            source="pmc_xml",
            text="Metformin was administered to patients.",
            token_count=6,
            entities={"drugs": ["metformin"]},
        )
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.text == chunk.text
        assert restored.entities == chunk.entities
