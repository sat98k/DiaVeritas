# =============================================================================
# src/preprocessing/xml_parser.py
#
# Parses PMC full-text XML into structured sections.
#
# PMC XML follows the JATS (Journal Article Tag Suite) schema.
# We extract:
#   - Title, authors, year, journal, DOI, abstract
#   - Body sections: Introduction, Methods, Results, Discussion, Conclusion
#
# Output: List of SectionBlock objects, each containing a section label
# and its raw text.
#
# Falls back gracefully when sections are missing — never invents content.
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from loguru import logger
from lxml import etree

from src.utils.text_utils import normalize_section_name, clean_text


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SectionBlock:
    """A named section extracted from a paper."""
    section: str          # Canonical section name (e.g., "Methods")
    raw_heading: str      # Original heading text from XML
    text: str             # Section text (cleaned)
    order: int            # Position of section in the paper


@dataclass
class ParsedPaper:
    """
    Structured output from parsing one paper's XML.

    All fields that may be absent use "Not reported" or empty defaults.
    """
    paper_id: str
    pmc_id: Optional[str]
    title: str
    authors: List[str]
    year: str
    journal: str
    doi: Optional[str]
    abstract: str
    sections: List[SectionBlock] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)
    source: str = "pmc_xml"


# ---------------------------------------------------------------------------
# JATS namespace handling
# ---------------------------------------------------------------------------

# PMC XML may or may not use explicit namespaces.
# We use namespace-agnostic XPath where possible.

def _strip_ns(tag: str) -> str:
    """Remove namespace prefix from an XML tag."""
    return re.sub(r"\{[^}]+\}", "", tag)


def _get_text_recursive(element) -> str:
    """
    Extract all text content from an element and its descendants,
    joining with spaces. Skips <table> and <fig> elements to avoid
    injecting tabular noise into running text.
    """
    skip_tags = {"table", "table-wrap", "fig", "graphic", "media", "supplementary-material"}
    parts: List[str] = []

    def _recurse(el):
        tag = _strip_ns(el.tag) if isinstance(el.tag, str) else ""
        if tag in skip_tags:
            return
        if el.text:
            parts.append(el.text.strip())
        for child in el:
            _recurse(child)
            if child.tail:
                parts.append(child.tail.strip())

    _recurse(element)
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# PMC XML parser
# ---------------------------------------------------------------------------

def parse_pmc_xml(xml_path: Path, paper_id: str) -> ParsedPaper:
    """
    Parse a PMC full-text XML file and return a ParsedPaper.

    Args:
        xml_path: Path to the .xml file.
        paper_id: The PMID used as the primary identifier.

    Returns:
        ParsedPaper with all available sections populated.
    """
    errors: List[str] = []

    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML {xml_path}: {e}")
        return ParsedPaper(
            paper_id=paper_id,
            pmc_id=None,
            title="Not reported",
            authors=["Not reported"],
            year="Unknown",
            journal="Not reported",
            doi=None,
            abstract="Not reported",
            parse_errors=[f"XML parse error: {e}"],
        )

    # ---- Title ----
    title = _extract_title(root) or "Not reported"

    # ---- Authors ----
    authors = _extract_authors(root) or ["Not reported"]

    # ---- Year ----
    year = _extract_year(root) or "Unknown"

    # ---- Journal ----
    journal = _extract_journal(root) or "Not reported"

    # ---- DOI ----
    doi = _extract_doi(root)

    # ---- PMC ID ----
    pmc_id = _extract_pmc_id(root)

    # ---- Abstract ----
    abstract = _extract_abstract(root) or "Not reported"

    # ---- Body sections ----
    sections = _extract_body_sections(root)
    if not sections and abstract and abstract != "Not reported":
        sections = [
            SectionBlock(
                section="Abstract",
                raw_heading="Abstract",
                text=clean_text(abstract),
                order=0,
            )
        ]

    paper = ParsedPaper(
        paper_id=paper_id,
        pmc_id=pmc_id,
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        abstract=clean_text(abstract),
        sections=sections,
        parse_errors=errors,
        source="pmc_xml",
    )

    section_names = [s.section for s in sections]
    logger.info(f"Parsed {xml_path.name}: {len(sections)} sections — {section_names}")
    return paper


# ---------------------------------------------------------------------------
# PubMed abstract XML parser (fallback)
# ---------------------------------------------------------------------------

def parse_pubmed_abstract_xml(xml_path: Path, paper_id: str) -> ParsedPaper:
    """
    Parse a PubMed abstract-only XML file.

    This gives structured metadata and abstract text but no body sections.
    The abstract is treated as the sole section.
    """
    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse abstract XML {xml_path}: {e}")
        return ParsedPaper(
            paper_id=paper_id,
            pmc_id=None,
            title="Not reported",
            authors=["Not reported"],
            year="Unknown",
            journal="Not reported",
            doi=None,
            abstract="Not reported",
            parse_errors=[f"XML parse error: {e}"],
            source="pubmed_abstract_xml",
        )

    title = _extract_title(root) or "Not reported"
    authors = _extract_authors(root) or ["Not reported"]
    year = _extract_year(root) or "Unknown"
    journal = _extract_journal(root) or "Not reported"
    doi = _extract_doi(root)
    pmc_id = _extract_pmc_id(root)
    abstract = _extract_abstract(root) or "Not reported"

    # Treat abstract as the only section
    abstract_section = SectionBlock(
        section="Abstract",
        raw_heading="Abstract",
        text=clean_text(abstract),
        order=0,
    )

    logger.info(f"Parsed abstract XML {xml_path.name}: abstract-only (no full text)")
    return ParsedPaper(
        paper_id=paper_id,
        pmc_id=pmc_id,
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        abstract=clean_text(abstract),
        sections=[abstract_section],
        source="pubmed_abstract_xml",
    )


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_title(root) -> Optional[str]:
    candidates = root.xpath("//*[local-name()='article-title']")
    if candidates:
        return clean_text(_get_text_recursive(candidates[0]))
    return None


def _extract_authors(root) -> List[str]:
    name_elements = root.xpath("//*[local-name()='contrib'][@contrib-type='author']")
    authors = []
    for contrib in name_elements:
        surname_el = contrib.xpath(".//*[local-name()='surname']")
        given_el = contrib.xpath(".//*[local-name()='given-names']")
        if surname_el:
            surname = surname_el[0].text or ""
            given = given_el[0].text if given_el else ""
            name = f"{surname}, {given}".strip(", ")
            if name:
                authors.append(name)
    return authors


def _extract_year(root) -> Optional[str]:
    # Try pub-date year first
    year_els = root.xpath("//*[local-name()='pub-date']//*[local-name()='year']")
    if year_els:
        return (year_els[0].text or "").strip() or None
    # Fallback: any year element
    year_els = root.xpath("//*[local-name()='year']")
    for el in year_els:
        text = (el.text or "").strip()
        if re.match(r"(19|20)\d{2}", text):
            return text
    return None


def _extract_journal(root) -> Optional[str]:
    candidates = root.xpath("//*[local-name()='journal-title']")
    if candidates:
        return (candidates[0].text or "").strip() or None
    candidates = root.xpath("//*[local-name()='abbrev-journal-title']")
    if candidates:
        return (candidates[0].text or "").strip() or None
    return None


def _extract_doi(root) -> Optional[str]:
    candidates = root.xpath("//*[local-name()='article-id'][@pub-id-type='doi']")
    if candidates:
        return (candidates[0].text or "").strip() or None
    return None


def _extract_pmc_id(root) -> Optional[str]:
    candidates = root.xpath("//*[local-name()='article-id'][@pub-id-type='pmc']")
    if candidates:
        raw = (candidates[0].text or "").strip()
        if raw:
            return f"PMC{raw}" if not raw.startswith("PMC") else raw
    return None


def _extract_abstract(root) -> Optional[str]:
    abstract_els = root.xpath("//*[local-name()='abstract']")
    if not abstract_els:
        # PubMed XML
        abstract_els = root.xpath(".//AbstractText")
    if not abstract_els:
        return None
    parts = []
    for el in abstract_els:
        text = _get_text_recursive(el).strip()
        if text:
            parts.append(text)
    return " ".join(parts) if parts else None


def _extract_body_sections(root) -> List[SectionBlock]:
    """
    Extract body sections from JATS XML.

    Handles nested <sec> elements. Flattens to top-level sections only
    (sub-sections are included in their parent's text).
    """
    body_els = root.xpath("//*[local-name()='body']")
    if not body_els:
        return []

    body = body_els[0]
    sections: List[SectionBlock] = []
    order = 0

    # Process top-level <sec> elements
    sec_els = body.xpath("./*[local-name()='sec']")

    if not sec_els:
        # Some papers have no <sec> wrapper — treat entire body as one block
        full_text = _get_text_recursive(body).strip()
        if full_text:
            sections.append(SectionBlock(
                section="Body",
                raw_heading="Body",
                text=clean_text(full_text),
                order=0,
            ))
        return sections

    for sec in sec_els:
        # Extract heading
        title_els = sec.xpath("./*[local-name()='title']")
        raw_heading = ""
        if title_els:
            raw_heading = _get_text_recursive(title_els[0]).strip()

        canonical = normalize_section_name(raw_heading) if raw_heading else "Unknown"

        # Get section text (all paragraphs + sub-sections)
        text = _get_text_recursive(sec).strip()
        # Remove the heading text from the start if it appears there
        if raw_heading and text.startswith(raw_heading):
            text = text[len(raw_heading):].strip()

        if not text:
            continue

        sections.append(SectionBlock(
            section=canonical,
            raw_heading=raw_heading,
            text=clean_text(text),
            order=order,
        ))
        order += 1

    return sections
