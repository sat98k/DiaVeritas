# =============================================================================
# src/preprocessing/pdf_parser.py
#
# PDF fallback parser using PyMuPDF (fitz).
#
# Used when:
#   - A paper has no PMC full-text XML available, AND
#   - A PDF file has been manually placed in data/raw/
#
# Limitations (documented, not hidden):
#   - PDF text extraction can be noisy (headers, footers, columns, tables)
#   - Section detection is heuristic-based, not guaranteed
#   - References section is stripped to reduce noise
#   - Table/figure content is not reliably filtered
#
# This parser produces the same ParsedPaper structure as xml_parser.py
# so downstream components need no special-casing.
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF parsing. Run: py -m pip install pymupdf")

from src.preprocessing.xml_parser import ParsedPaper, SectionBlock
from src.utils.text_utils import clean_text, normalize_section_name


# ---------------------------------------------------------------------------
# Section heading detection heuristics
# ---------------------------------------------------------------------------

# Patterns that commonly indicate a section heading in academic PDFs.
# These are intentionally conservative to avoid false positives.
_SECTION_HEADING_PATTERNS = [
    re.compile(r"^(Abstract|Introduction|Background|Methods?|Materials? and Methods?|"
               r"Results?|Findings?|Discussion|Conclusions?|Acknowledgem?ents?|"
               r"References?|Supplementary|Appendix|Funding|Limitations?|"
               r"Study Design|Participants?|Patients?|Outcomes?)\s*$",
               re.IGNORECASE),
    # Numbered headings like "1. Introduction", "2 Methods"
    re.compile(r"^\d+\.?\s+(Abstract|Introduction|Background|Methods?|Results?|"
               r"Discussion|Conclusions?|References?)\s*$", re.IGNORECASE),
]


def _is_section_heading(line: str, font_size: Optional[float] = None,
                        body_font_size: Optional[float] = None) -> bool:
    """
    Heuristic check for whether a line is a section heading.

    Checks:
    1. Pattern match against known section names
    2. Font size significantly larger than body text (if available)
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    for pattern in _SECTION_HEADING_PATTERNS:
        if pattern.match(stripped):
            return True
    # Font size heuristic: heading if >20% larger than body
    if font_size is not None and body_font_size is not None and body_font_size > 0:
        if font_size >= body_font_size * 1.2 and len(stripped.split()) <= 8:
            return True
    return False


def _estimate_body_font_size(blocks: list) -> float:
    """Estimate the most common font size (proxy for body text size)."""
    sizes: List[float] = []
    for block in blocks:
        if block.get("type") != 0:  # type 0 = text
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 0))
    if not sizes:
        return 11.0  # fallback
    # Mode-ish: round to nearest 0.5 and pick most common
    rounded = [round(s * 2) / 2 for s in sizes]
    return max(set(rounded), key=rounded.count)


# ---------------------------------------------------------------------------
# Main PDF parser
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path, paper_id: str, metadata: Optional[dict] = None) -> ParsedPaper:
    """
    Parse a PDF file into a ParsedPaper with section blocks.

    Args:
        pdf_path: Path to the PDF file.
        paper_id: PMID or other identifier.
        metadata: Optional pre-fetched metadata dict (from corpus registry).
                  If provided, title/authors/year/journal are taken from here
                  rather than guessed from the PDF.

    Returns:
        ParsedPaper. Sections will be heuristic-based.
        Missing metadata is set to "Not reported" / "Unknown".
    """
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return _empty_paper(paper_id, f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error(f"Cannot open PDF {pdf_path}: {e}")
        return _empty_paper(paper_id, f"PDF open error: {e}")

    # Use metadata from registry if available
    if metadata:
        title = metadata.get("title", "Not reported")
        authors = metadata.get("authors", ["Not reported"])
        year = metadata.get("year", "Unknown")
        journal = metadata.get("journal", "Not reported")
        doi = metadata.get("doi")
        pmc_id = metadata.get("pmc_id")
        abstract = metadata.get("abstract", "")
    else:
        title = "Not reported"
        authors = ["Not reported"]
        year = "Unknown"
        journal = "Not reported"
        doi = None
        pmc_id = None
        abstract = ""

    # Extract full text with block structure for font-size heuristics
    pages_text: List[Tuple[str, Optional[float]]] = []  # (line_text, font_size)
    all_blocks = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        all_blocks.extend(blocks)

    body_font_size = _estimate_body_font_size(all_blocks)

    raw_lines: List[Tuple[str, float]] = []  # (text, font_size)
    for block in all_blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = ""
            line_size = 0.0
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                line_size = span.get("size", 0.0)
            line_text = line_text.strip()
            if line_text:
                raw_lines.append((line_text, line_size))

    doc.close()

    # ---- Segment into sections ----
    sections = _segment_into_sections(raw_lines, body_font_size)

    # If we got an abstract from metadata, use it; otherwise look in sections
    if not abstract:
        for sec in sections:
            if sec.section == "Abstract":
                abstract = sec.text
                break

    # Strip References section to reduce noise
    sections = [s for s in sections if s.section != "References"]

    logger.info(
        f"Parsed PDF {pdf_path.name}: {len(sections)} sections "
        f"(body font ~{body_font_size:.1f}pt)"
    )

    return ParsedPaper(
        paper_id=paper_id,
        pmc_id=pmc_id,
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        abstract=clean_text(abstract),
        sections=sections,
        parse_errors=[],
        source="pdf_fallback",
    )


def _segment_into_sections(
    raw_lines: List[Tuple[str, float]],
    body_font_size: float,
) -> List[SectionBlock]:
    """
    Group raw lines into sections using heading detection heuristics.
    """
    sections: List[SectionBlock] = []
    current_heading = "Unknown"
    current_raw_heading = ""
    current_lines: List[str] = []
    order = 0

    def _flush():
        nonlocal order
        text = clean_text(" ".join(current_lines))
        if text:
            sections.append(SectionBlock(
                section=normalize_section_name(current_heading),
                raw_heading=current_raw_heading,
                text=text,
                order=order,
            ))
            order += 1

    for line_text, font_size in raw_lines:
        if _is_section_heading(line_text, font_size, body_font_size):
            _flush()
            current_heading = line_text.strip()
            current_raw_heading = line_text.strip()
            current_lines = []
        else:
            current_lines.append(line_text)

    _flush()  # Don't forget the last section

    return sections


def _empty_paper(paper_id: str, error: str) -> ParsedPaper:
    return ParsedPaper(
        paper_id=paper_id,
        pmc_id=None,
        title="Not reported",
        authors=["Not reported"],
        year="Unknown",
        journal="Not reported",
        doi=None,
        abstract="Not reported",
        sections=[],
        parse_errors=[error],
        source="pdf_fallback",
    )
