# =============================================================================
# scripts/build_index.py
#
# Offline index-building script for DiaVeritas.
#
# Usage:
#   py scripts/build_index.py --step ingest       # Download + parse + chunk
#   py scripts/build_index.py --step index        # Embed + store in vector DB + BM25
#   py scripts/build_index.py --step all          # Both steps
#   py scripts/build_index.py --step ingest --max-papers 20
#
# This script is the main entry point for the offline pipeline.
# It is designed to be run once (or re-run to refresh the corpus).
#
# NOTE: Large model downloads happen on first run of --step index.
# Expected download: ~500MB for the embedding model.
# =============================================================================

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import click
from loguru import logger

from src.config import settings


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--step",
    type=click.Choice(["ingest", "index", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help=(
        "ingest: download + parse + chunk. "
        "index: embed + vector store + BM25. "
        "all: both steps."
    ),
)
@click.option(
    "--max-papers",
    default=50,
    show_default=True,
    type=int,
    help="Maximum number of PubMed papers to download (ingest step).",
)
@click.option(
    "--min-year",
    default=2010,
    show_default=True,
    type=int,
    help="Minimum publication year for PubMed search (ingest step).",
)
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help=(
        "Clear existing chunks.jsonl and vector store before rebuilding. "
        "IMPORTANT: re-running without --fresh appends to the existing chunks.jsonl, "
        "which duplicates the BM25 corpus (the vector store is unaffected due to ID "
        "deduplication). Always use --fresh when rebuilding from scratch."
    ),
)
@click.option(
    "--chunk-file",
    default=None,
    type=click.Path(),
    help="Path to an existing chunks.jsonl file (skip ingestion, go straight to index).",
)
@click.option(
    "--log-level",
    default=settings.log_level,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Log verbosity.",
)
def main(step, max_papers, min_year, fresh, chunk_file, log_level):
    """
    DiaVeritas offline index builder.

    Re-running without --fresh APPENDS to data/processed/chunks.jsonl.
    This causes BM25 corpus duplication (the vector store uses ID deduplication
    and is unaffected). Use --fresh to rebuild cleanly from scratch.
    """
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    logger.info(f"DiaVeritas build_index — step={step}, max_papers={max_papers}")

    if step in ("ingest", "all"):
        run_ingest(max_papers=max_papers, min_year=min_year, fresh=fresh)

    if step in ("index", "all"):
        chunks_path = Path(chunk_file) if chunk_file else Path(settings.processed_dir) / "chunks.jsonl"
        run_index(chunks_path=chunks_path, fresh=fresh)

    logger.info("build_index complete.")


# ---------------------------------------------------------------------------
# Step 1: Ingest
# ---------------------------------------------------------------------------

def run_ingest(max_papers: int, min_year: int, fresh: bool) -> None:
    """
    Download corpus from PubMed/PMC, parse XML/PDF, chunk, extract entities.
    Saves: data/metadata/corpus_registry.json + data/processed/chunks.jsonl
    """
    from src.ingestion.pubmed_fetcher import build_corpus
    from src.ingestion.corpus_registry import add_papers, load_registry
    from src.preprocessing.xml_parser import parse_pmc_xml, parse_pubmed_abstract_xml
    from src.preprocessing.pdf_parser import parse_pdf
    from src.preprocessing.chunker import chunk_paper, save_chunks, clear_chunk_file
    from src.preprocessing.entity_extractor import enrich_chunks_with_entities

    chunks_out = Path(settings.processed_dir) / "chunks.jsonl"

    if fresh:
        logger.info("--fresh: clearing existing chunk file and registry.")
        clear_chunk_file(chunks_out)
        registry_file = Path(settings.metadata_dir) / "corpus_registry.json"
        if registry_file.exists():
            registry_file.unlink()

    # 1. Download corpus
    logger.info(f"Step 1/4: Downloading up to {max_papers} T2D papers from PubMed...")
    records = build_corpus(max_papers=max_papers, min_year=min_year)

    if not records:
        logger.error("No papers downloaded. Check NCBI credentials and network.")
        return

    # 2. Save to registry
    logger.info(f"Step 2/4: Saving {len(records)} papers to corpus registry...")
    add_papers(records)

    # 3. Parse each paper
    logger.info(f"Step 3/4: Parsing {len(records)} papers...")
    all_chunks = []

    for rec in records:
        paper_id = rec.paper_id
        study_type = rec.study_type

        parsed = None

        # Try PMC XML first
        if rec.local_xml_path and rec.source == "pmc_xml":
            xml_path = Path(rec.local_xml_path)
            if xml_path.exists():
                parsed = parse_pmc_xml(xml_path, paper_id)

        # Try PubMed abstract XML
        elif rec.local_xml_path and rec.source == "pubmed_abstract_xml":
            xml_path = Path(rec.local_xml_path)
            if xml_path.exists():
                parsed = parse_pubmed_abstract_xml(xml_path, paper_id)

        # Try PDF fallback
        elif rec.local_pdf_path:
            pdf_path = Path(rec.local_pdf_path)
            if pdf_path.exists():
                parsed = parse_pdf(pdf_path, paper_id, metadata=rec.to_dict())

        if parsed is None:
            logger.warning(f"No parseable file for paper {paper_id}. Skipping.")
            continue

        if parsed.parse_errors:
            logger.warning(f"Parse errors for {paper_id}: {parsed.parse_errors}")

        # Chunk
        chunks = chunk_paper(parsed, study_type=study_type)
        all_chunks.extend(chunks)

    # 4. Entity extraction
    logger.info(f"Step 4/4: Extracting entities from {len(all_chunks)} chunks...")
    enrich_chunks_with_entities(all_chunks)

    # Save chunks
    save_chunks(all_chunks, chunks_out)
    logger.info(
        f"Ingest complete: {len(records)} papers → {len(all_chunks)} chunks "
        f"→ {chunks_out}"
    )


# ---------------------------------------------------------------------------
# Step 2: Index
# ---------------------------------------------------------------------------

def run_index(chunks_path: Path, fresh: bool) -> None:
    """
    Load chunks, embed them, store in ChromaDB, and build BM25 index.
    """
    from src.preprocessing.chunker import load_chunks
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_index import BM25Index

    logger.info(f"Loading chunks from {chunks_path}...")
    chunks = load_chunks(chunks_path)

    if not chunks:
        logger.error(
            "No chunks found. Run --step ingest first, or specify --chunk-file."
        )
        return

    logger.info(f"Loaded {len(chunks)} chunks.")

    # Embedding model (downloads ~420MB on first run)
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    logger.info("(First run will download ~420MB — this may take a few minutes)")
    embedder = Embedder()

    # Vector store
    vector_store = VectorStore(fresh=fresh)

    # Check what's already indexed
    existing_ids = set(vector_store.get_all_ids())
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if not new_chunks:
        logger.info("All chunks already indexed in vector store.")
    else:
        logger.info(f"Embedding and indexing {len(new_chunks)} new chunks...")
        texts = [c.text for c in new_chunks]
        embeddings = embedder.encode(texts)
        vector_store.add_chunks(new_chunks, embeddings)
        logger.info(f"Vector store: {len(new_chunks)} chunks added.")

    # BM25 index (always rebuilt from all chunks — fast, in-memory)
    logger.info("Building BM25 index...")
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
    bm25_index.save(bm25_path)
    logger.info(f"BM25 index saved to {bm25_path}")

    logger.info("Index step complete.")


if __name__ == "__main__":
    main()
