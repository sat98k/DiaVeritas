import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import click
from loguru import logger

from src.config import settings
from src.ingestion.pubmed_fetcher import build_domain_diversified_corpus, LANDMARK_T2D_PMIDS, DOMAIN_QUERIES
from src.ingestion.corpus_registry import add_papers
from src.preprocessing.xml_parser import parse_pmc_xml, parse_pubmed_abstract_xml
from src.preprocessing.pdf_parser import parse_pdf
from src.preprocessing.chunker import chunk_paper, save_chunks, load_chunks
from src.preprocessing.entity_extractor import enrich_chunks_with_entities
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index


@click.command()
@click.option(
    "--target-papers",
    default=1000,
    show_default=True,
    type=int,
    help="Target number of clinical papers across all 5 domains (e.g., 500, 1000, 2000).",
)
@click.option(
    "--min-year",
    default=2000,
    show_default=True,
    type=int,
    help="Minimum publication year for literature search.",
)
@click.option(
    "--fresh/--append",
    default=True,
    show_default=True,
    help="Clear existing index and rebuild from scratch (recommended).",
)
def main(target_papers: int, min_year: int, fresh: bool):
    """
    Build a comprehensive, domain-diversified Type 2 Diabetes clinical dataset (1000-2000 papers).
    Covers:
      1. Nutrition & Diet (caffeine, coffee, sugar, Mediterranean, fiber)
      2. Pharmacotherapy (Metformin, GLP-1, SGLT2i, tirzepatide, semaglutide)
      3. Exercise & Lifestyle (aerobic, resistance, HIIT, sedentary)
      4. Complications & CVD/CKD (MACE, heart failure, renal outcomes)
      5. Prediabetes & Remission (DPP, lifestyle prevention, remission)
    """
    logger.info("=" * 70)
    logger.info(f"DiaVeritas: Building Multi-Pillar Clinical Corpus ({target_papers} target papers)")
    logger.info(f"Clinical Pillars: {list(DOMAIN_QUERIES.keys())}")
    logger.info(f"Landmark Foundations: {len(LANDMARK_T2D_PMIDS)} trials pre-seeded")
    logger.info("=" * 70)

    raw_dir = Path(settings.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunks_out = Path(settings.processed_dir) / "chunks.jsonl"
    bm25_out = Path(settings.processed_dir) / "bm25_index.pkl"

    # 1. Ingest across all 5 domains
    logger.info(f"Phase 1/4: Downloading balanced multi-domain literature...")
    records = build_domain_diversified_corpus(
        total_papers=target_papers,
        raw_dir=raw_dir,
        min_year=min_year,
    )

    add_papers(records)
    logger.info(f"Phase 1 complete: {len(records)} papers recorded in registry.")

    # 2. Parse & Chunk
    logger.info(f"Phase 2/4: Parsing and segmenting into boundary-aware chunks...")
    all_chunks = []
    for rec in records:
        paper_id = rec.paper_id
        study_type = rec.study_type
        parsed = None

        if rec.local_xml_path and rec.source == "pmc_xml":
            xml_path = Path(rec.local_xml_path)
            if xml_path.exists():
                parsed = parse_pmc_xml(xml_path, paper_id)
        elif rec.local_xml_path and rec.source == "pubmed_abstract_xml":
            xml_path = Path(rec.local_xml_path)
            if xml_path.exists():
                parsed = parse_pubmed_abstract_xml(xml_path, paper_id)
        elif rec.local_pdf_path:
            pdf_path = Path(rec.local_pdf_path)
            if pdf_path.exists():
                parsed = parse_pdf(pdf_path, paper_id, metadata=rec.to_dict())

        if parsed:
            chunks = chunk_paper(parsed, study_type=study_type)
            all_chunks.extend(chunks)

    # 3. Entity Extraction & Save Chunks
    logger.info(f"Phase 3/4: Extracting clinical entities from {len(all_chunks)} chunks...")
    enrich_chunks_with_entities(all_chunks)
    save_chunks(all_chunks, chunks_out)
    logger.info(f"Saved {len(all_chunks)} chunks to {chunks_out}")

    # 4. Dense & Sparse Indexing
    logger.info(f"Phase 4/4: Building Vector Store (ChromaDB) and Sparse Index (BM25)...")
    embedder = Embedder()
    vector_store = VectorStore(fresh=fresh)
    
    logger.info(f"Embedding {len(all_chunks)} chunks with S-PubMedBert...")
    texts = [c.text for c in all_chunks]
    embeddings = embedder.encode(texts)
    vector_store.add_chunks(all_chunks, embeddings)

    logger.info("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(all_chunks)
    bm25.save(bm25_out)

    logger.info("=" * 70)
    logger.info("✅ Comprehensive Clinical Dataset Build Successfully Complete!")
    logger.info(f"  • Total Papers: {len(records)}")
    logger.info(f"  • Total Chunks: {len(all_chunks)}")
    logger.info(f"  • Vector Store: ChromaDB ({settings.vector_store_path})")
    logger.info(f"  • BM25 Index:   {bm25_out}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
