# =============================================================================
# src/ingestion/corpus_registry.py
#
# Persistent registry of all papers in the corpus.
#
# Maintains a single JSON file at data/metadata/corpus_registry.json
# that maps paper_id → PaperRecord metadata.
#
# This is the single source of truth for what is in the corpus.
# The preprocessing and retrieval stages read from this registry.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

from loguru import logger

from src.config import settings
from src.ingestion.pubmed_fetcher import PaperRecord


REGISTRY_FILE = Path(settings.metadata_dir) / "corpus_registry.json"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_registry() -> Dict[str, dict]:
    """
    Load the corpus registry from disk.
    Returns an empty dict if the registry does not exist yet.
    """
    if not REGISTRY_FILE.exists():
        logger.debug("Corpus registry does not exist yet; returning empty registry.")
        return {}
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"Loaded corpus registry: {len(data)} papers")
    return data


def save_registry(registry: Dict[str, dict]) -> None:
    """Persist the registry to disk, creating parent directories if needed."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved corpus registry: {len(registry)} papers → {REGISTRY_FILE}")


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------

def add_papers(records: List[PaperRecord]) -> Dict[str, dict]:
    """
    Add or update PaperRecord entries in the registry.

    Existing entries are updated in-place (e.g., if local_xml_path is
    now populated after a download).

    Returns the updated registry.
    """
    registry = load_registry()
    added = 0
    updated = 0
    for rec in records:
        d = rec.to_dict()
        if rec.paper_id not in registry:
            added += 1
        else:
            updated += 1
        registry[rec.paper_id] = d

    save_registry(registry)
    logger.info(f"Registry updated: {added} added, {updated} updated.")
    return registry


def get_paper(paper_id: str) -> Optional[dict]:
    """Retrieve a single paper's metadata from the registry."""
    registry = load_registry()
    return registry.get(paper_id)


def list_papers(source_filter: Optional[str] = None) -> List[dict]:
    """
    List all papers in the registry.

    Args:
        source_filter: If provided, only return papers with matching source
                       (e.g., 'pmc_xml', 'pubmed_abstract_xml').
    """
    registry = load_registry()
    papers = list(registry.values())
    if source_filter:
        papers = [p for p in papers if p.get("source") == source_filter]
    return papers


def list_paper_ids() -> List[str]:
    """Return all paper IDs currently in the registry."""
    return list(load_registry().keys())


def registry_summary() -> Dict[str, int]:
    """
    Return a summary count of papers by source type.
    Useful for quick status checks.
    """
    registry = load_registry()
    summary: Dict[str, int] = {}
    for paper in registry.values():
        src = paper.get("source", "unknown")
        summary[src] = summary.get(src, 0) + 1
    summary["total"] = len(registry)
    return summary
