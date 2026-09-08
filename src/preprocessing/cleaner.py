# =============================================================================
# src/preprocessing/cleaner.py
#
# Text cleaning utilities specific to biomedical literature.
#
# Applied AFTER XML/PDF parsing, BEFORE chunking.
# Removes common artifacts introduced by PDF extraction and XML parsing
# without modifying the scientific content.
#
# Design note: cleaning is intentionally conservative.
# We do NOT:
#   - Remove numbers or special characters that may be clinically meaningful
#   - Normalize drug names or expand abbreviations (that is entity extraction's job)
#   - Discard text we cannot confidently classify as non-content
# =============================================================================

from __future__ import annotations

import re
from typing import List

from src.utils.text_utils import clean_text


# ---------------------------------------------------------------------------
# Biomedical-specific noise patterns
# ---------------------------------------------------------------------------

# Lines or spans that are almost certainly headers/footers/page numbers
_NOISE_PATTERNS = [
    re.compile(r"^\d+$"),                          # Page numbers
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE),
    re.compile(r"^(©|Copyright|All rights reserved)", re.IGNORECASE),
    re.compile(r"^doi:\s*10\.\d{4,}", re.IGNORECASE),  # DOI lines
    re.compile(r"^https?://\S+$"),                 # Standalone URLs
    re.compile(r"^\[?\d+\]?\s*$"),                 # Reference number only
    re.compile(r"^Received:.*Accepted:", re.IGNORECASE),  # Journal metadata
    re.compile(r"^Published online:", re.IGNORECASE),
    re.compile(r"^Open Access$", re.IGNORECASE),
]

# Inline noise patterns to replace
_INLINE_REPLACEMENTS = [
    # Citation numbers like [1], [2,3], [1-5]
    (re.compile(r"\[\d+(?:[-,]\d+)*\]"), " "),
    # Superscript reference numbers (often appear as isolated digits in PDF)
    (re.compile(r"\s\d{1,3}(?=\s)"), " "),  # only very short isolated numbers
    # Hyphenation artifacts (line-break hyphens)
    (re.compile(r"(\w)-\s*\n\s*(\w)"), r"\1\2"),
    # Multiple spaces
    (re.compile(r" {2,}"), " "),
]


def clean_section_text(text: str) -> str:
    """
    Clean text from a single section.

    Steps:
    1. Apply base cleaning (unicode, whitespace, dashes, quotes)
    2. Remove line-level noise (headers, footers, page numbers)
    3. Apply inline replacements
    4. Final whitespace normalization

    This is the primary cleaning function called per section.
    """
    # Step 1: Base cleaning
    text = clean_text(text)

    # Step 2: Line-level noise removal
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(pat.search(stripped) for pat in _NOISE_PATTERNS):
            continue
        cleaned_lines.append(stripped)
    text = " ".join(cleaned_lines)

    # Step 3: Inline replacements
    for pattern, replacement in _INLINE_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Step 4: Final normalization
    text = re.sub(r" {2,}", " ", text).strip()

    return text


def remove_references_section(text: str) -> str:
    """
    Attempt to remove the References section from body text.

    Used when sections are not properly split and the references
    appear at the end of a merged text block.

    Conservative: only removes from a clear "References" heading
    at the start of a line to end-of-text.
    """
    pattern = re.compile(
        r"\n\s*(References?|Bibliography)\s*\n.+$",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub("", text).strip()


def filter_short_chunks(texts: List[str], min_tokens: int = 20) -> List[str]:
    """
    Filter out chunks that are too short to be meaningful evidence.

    Chunks under min_tokens words are discarded — they are likely
    artifact fragments, not substantive evidence passages.
    """
    return [t for t in texts if len(t.split()) >= min_tokens]
