# =============================================================================
# src/utils/text_utils.py
#
# Shared text helpers used across the pipeline.
# Keeps sentence splitting and tokenization consistent everywhere.
# =============================================================================

from __future__ import annotations

import re
import unicodedata
from typing import List


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a lightweight regex approach.

    This is intentionally simple and fast.  For biomedical text we rely on
    end-of-sentence punctuation patterns rather than a full ML model, because
    the overhead of an ML splitter is not justified at this stage and the
    corpus is already relatively clean scientific prose.

    Edge cases handled:
    - Abbreviations like e.g., i.e., et al., vs., Fig., Dr., Prof.
    - Decimal numbers (3.14)
    - References like [1], (Smith et al., 2020)
    """
    # Protect common abbreviations by replacing their periods temporarily
    _abbrevs = [
        r"(?<!\w)(e\.g|i\.e|et al|vs|fig|dr|prof|mr|ms|mrs|jr|sr|no|vol|pp|approx|dept|eq|ref)\.",
    ]
    protected = text
    for pattern in _abbrevs:
        protected = re.sub(pattern, lambda m: m.group(0).replace(".", "<DOT>"), protected, flags=re.IGNORECASE)

    # Protect decimal numbers (e.g. 3.14, 0.05, p<0.001)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    # Split on sentence boundaries: period / ! / ? followed by space + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'(\[])", protected)

    # Restore protected dots and strip
    result = []
    for s in sentences:
        s = s.replace("<DOT>", ".").strip()
        if s:
            result.append(s)

    return result if result else [text.strip()]


# ---------------------------------------------------------------------------
# Rough token counter (whitespace-based)
# ---------------------------------------------------------------------------

def count_tokens_approx(text: str) -> int:
    """
    Approximate token count using whitespace split.

    This is intentionally a rough estimate — good enough for chunking
    decisions without loading a full tokenizer.  A 300–500 token target
    corresponds to roughly 220–380 words in typical biomedical prose.

    For precise counts, use a HuggingFace tokenizer directly.
    """
    return len(text.split())


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, tabs, and newlines into single spaces."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode to NFKC form.

    Handles common issues from PDF/XML extraction:
    ligatures (ﬁ → fi), special dashes (– → -), fancy quotes, etc.
    """
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline: unicode → whitespace → strip.
    Applied to raw extracted text before chunking.
    """
    text = normalize_unicode(text)
    # Replace various dash types with ASCII hyphen
    text = re.sub(r"[–—−]", "-", text)
    # Replace fancy quotes (using Unicode escapes for cross-platform safety)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    # Remove null bytes and control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Section name normalization
# ---------------------------------------------------------------------------

_SECTION_MAP = {
    # Abstract variants
    "abstract": "Abstract",
    "summary": "Abstract",
    # Introduction
    "introduction": "Introduction",
    "background": "Introduction",
    "intro": "Introduction",
    # Methods
    "methods": "Methods",
    "method": "Methods",
    "materials and methods": "Methods",
    "methodology": "Methods",
    "study design": "Methods",
    "participants": "Methods",
    "patients": "Methods",
    "subjects": "Methods",
    # Results
    "results": "Results",
    "findings": "Results",
    "outcomes": "Results",
    # Discussion
    "discussion": "Discussion",
    "interpretation": "Discussion",
    # Conclusion
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "concluding remarks": "Conclusion",
    # Other
    "references": "References",
    "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
    "funding": "Acknowledgements",
    "supplementary": "Supplementary",
    "appendix": "Supplementary",
    "figure": "Figures/Tables",
    "table": "Figures/Tables",
}


def normalize_section_name(raw: str) -> str:
    """
    Map raw section heading text to a canonical section label.

    Returns the canonical label if recognized, otherwise returns
    the original (title-cased) string.
    """
    if not raw:
        return "Unknown"
    key = raw.strip().lower()
    return _SECTION_MAP.get(key, raw.strip().title())


# ---------------------------------------------------------------------------
# Chunk overlap helpers
# ---------------------------------------------------------------------------

def compute_overlap_sentences(sentences: List[str], overlap_tokens: int) -> List[str]:
    """
    Return the trailing sentences from `sentences` whose combined token count
    is approximately `overlap_tokens`.  Used to prepend overlap to the next chunk.
    """
    if overlap_tokens <= 0 or not sentences:
        return []
    collected = []
    total = 0
    for sent in reversed(sentences):
        t = count_tokens_approx(sent)
        if total + t > overlap_tokens * 1.5:  # allow 50% slack
            break
        collected.insert(0, sent)
        total += t
    return collected
