# =============================================================================
# src/preprocessing/chunker.py
#
# Section-aware, sentence-boundary chunking.
#
# Core design decisions:
#   - Chunks are always sentence-aligned (never cut mid-sentence)
#   - Section identity is preserved per chunk (Methods chunks stay Methods)
#   - Chunk size and overlap are configurable via settings (config.py)
#   - Output is a flat list of Chunk objects (JSONL-serializable)
#
# Chunking parameters are EXPERIMENTAL starting points, not optimal values.
# They are intentionally easy to modify in config.py.
#
# Output schema per chunk (matches vector store metadata):
# {
#   "chunk_id": "37123456_Methods_002",
#   "paper_id": "37123456",
#   "pmc_id": "PMC9876543",
#   "title": "...",
#   "authors": ["Smith J, ..."],
#   "year": "2021",
#   "journal": "Diabetes Care",
#   "section": "Methods",
#   "chunk_index": 2,
#   "study_type": "RCT",
#   "source": "pmc_xml",
#   "text": "...",
#   "token_count": 387,
#   "entities": {}        # populated by entity_extractor
# }
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from src.config import settings
from src.preprocessing.xml_parser import ParsedPaper, SectionBlock
from src.utils.text_utils import split_sentences, count_tokens_approx, compute_overlap_sentences
from src.preprocessing.cleaner import clean_section_text, filter_short_chunks


# ---------------------------------------------------------------------------
# Chunk data structure
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single evidence chunk from one paper's section.

    Every field is always populated — missing values use "Not reported"
    or "Unknown", never None (for JSON serialization cleanliness).
    """
    chunk_id: str               # Unique: "<paper_id>_<section>_<index>"
    paper_id: str
    pmc_id: str                 # "Unknown" if unavailable
    title: str
    authors: List[str]
    year: str
    journal: str
    doi: str                    # "Not reported" if unavailable
    section: str
    chunk_index: int
    study_type: str             # "Not reported" if unavailable
    source: str                 # "pmc_xml" | "pdf_fallback" | "pubmed_abstract_xml"
    text: str
    token_count: int
    entities: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(**d)


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def chunk_section(
    section: SectionBlock,
    paper: ParsedPaper,
    study_type: str,
    chunk_size_tokens: Optional[int] = None,
    overlap_percent: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk a single section into token-bounded, sentence-aligned chunks.

    Args:
        section: The section to chunk.
        paper: The parent ParsedPaper (for metadata).
        study_type: Study type string from registry (e.g., "RCT").
        chunk_size_tokens: Target chunk size in tokens.
                           Defaults to settings.chunk_size_tokens.
        overlap_percent: Overlap between chunks as % of chunk_size.
                         Defaults to settings.chunk_overlap_percent.

    Returns:
        List of Chunk objects for this section.
    """
    chunk_size = chunk_size_tokens or settings.chunk_size_tokens
    overlap_pct = overlap_percent if overlap_percent is not None else settings.chunk_overlap_percent
    overlap_tokens = int(chunk_size * overlap_pct / 100)

    # Clean the section text
    text = clean_section_text(section.text)
    if not text:
        return []

    # Split into sentences
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_tokens = 0
    overlap_buffer: List[str] = []  # sentences to prepend to next chunk
    chunk_index = 0

    def _make_chunk(sents: List[str], idx: int) -> Optional[Chunk]:
        joined = " ".join(sents).strip()
        token_count = count_tokens_approx(joined)
        # Filter chunks that are too short (artifact fragments)
        if token_count < 20:
            return None

        # Build safe chunk_id: replace spaces in section name
        section_safe = section.section.replace(" ", "_").replace("/", "_")
        chunk_id = f"{paper.paper_id}_{section_safe}_{idx:03d}"

        return Chunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            pmc_id=paper.pmc_id or "Unknown",
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            journal=paper.journal,
            doi=paper.doi or "Not reported",
            section=section.section,
            chunk_index=idx,
            study_type=study_type,
            source=paper.source,
            text=joined,
            token_count=token_count,
        )

    for sent in sentences:
        sent_tokens = count_tokens_approx(sent)

        # If adding this sentence would exceed the chunk size, flush
        if current_tokens + sent_tokens > chunk_size and current_sentences:
            chunk = _make_chunk(current_sentences, chunk_index)
            if chunk:
                chunks.append(chunk)
                chunk_index += 1

            # Compute overlap: take trailing sentences from current chunk
            overlap_buffer = compute_overlap_sentences(current_sentences, overlap_tokens)
            current_sentences = overlap_buffer[:]
            current_tokens = sum(count_tokens_approx(s) for s in current_sentences)

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Flush the last chunk
    if current_sentences:
        chunk = _make_chunk(current_sentences, chunk_index)
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_paper(
    paper: ParsedPaper,
    study_type: str = "Not reported",
    chunk_size_tokens: Optional[int] = None,
    overlap_percent: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk all sections of a paper.

    Skips References and Acknowledgements sections (low evidence value).
    Returns a flat list of all chunks across all sections.
    """
    skip_sections = {"References", "Acknowledgements", "Figures/Tables"}
    all_chunks: List[Chunk] = []

    for section in paper.sections:
        if section.section in skip_sections:
            logger.debug(f"Skipping section '{section.section}' for paper {paper.paper_id}")
            continue

        section_chunks = chunk_section(
            section=section,
            paper=paper,
            study_type=study_type,
            chunk_size_tokens=chunk_size_tokens,
            overlap_percent=overlap_percent,
        )
        all_chunks.extend(section_chunks)

    logger.info(
        f"Chunked paper {paper.paper_id}: "
        f"{len(paper.sections)} sections → {len(all_chunks)} chunks"
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_chunks(chunks: List[Chunk], output_path: Path) -> None:
    """
    Save chunks to a JSONL file (one JSON object per line).

    Appends to existing file if it exists, allowing incremental builds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.debug(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path) -> List[Chunk]:
    """
    Load chunks from a JSONL file.

    Returns an empty list if the file does not exist.
    """
    if not input_path.exists():
        logger.warning(f"Chunk file not found: {input_path}")
        return []
    chunks = []
    seen_ids = set()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    c = Chunk.from_dict(json.loads(line))
                    if c.chunk_id not in seen_ids:
                        seen_ids.add(c.chunk_id)
                        chunks.append(c)
                except Exception as e:
                    logger.warning(f"Skipping malformed chunk line: {e}")
    logger.info(f"Loaded {len(chunks)} unique chunks from {input_path}")
    return chunks


def clear_chunk_file(output_path: Path) -> None:
    """Delete the chunk file to start fresh (used before a full re-index)."""
    if output_path.exists():
        output_path.unlink()
        logger.info(f"Cleared chunk file: {output_path}")
