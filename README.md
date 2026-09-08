# =============================================================================
# DiaVeritas
# RAG-Based Medical Literature Contradiction Detection and Evidence Analysis
# System for Type 2 Diabetes Treatment Literature
# =============================================================================

## Overview

DiaVeritas is a research-oriented NLP system that analyzes Type 2 Diabetes treatment literature and identifies supporting, contradicting, and context-dependent evidence before generating a citation-grounded answer.

**This system is NOT:**
- A medical diagnosis system
- A treatment recommendation system
- A clinical decision-support system
- A generic medical chatbot

**Core principle:** Retrieve → Analyze → Compare → Synthesize

---

## Quick Start

### 1. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 2. Install scispaCy model (optional but recommended)

```bash
py -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz
```

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env and add your API keys (at minimum, GROQ_API_KEY for LLM synthesis)
```

### 4. Build the corpus index

```bash
# Download 30 T2D papers from PubMed, parse, chunk, embed, and index
py scripts/build_index.py --step all --max-papers 30
```

### 5. Launch the UI

```bash
py -m streamlit run ui/app.py
```

---

## Pipeline Overview

```
OFFLINE:
PubMed/PMC → XML/PDF Parsing → Cleaning → Chunking → Entity Extraction
                                                ↓
                                        Biomedical Embeddings
                                        Vector DB (ChromaDB)

ONLINE:
User Question
    ↓ Query Normalization
    ↓ Dense Retrieval + BM25
    ↓ RRF Hybrid Fusion
    ↓ Cross-Encoder Reranking
    ↓ Structured Claim Extraction
    ↓ Biomedical NLI (Entailment / Contradiction / Neutral)
    ↓ Contextual Contradiction Analysis
    ↓ Evidence Relationship Derivation
    ↓ Evidence Status (SUPPORTED / REFUTED / INCONCLUSIVE)
    ↓ LLM Synthesis
    ↓ Citation-Grounded Answer
```

---

## Project Structure

```
DiaVeritas/
├── src/
│   ├── config.py                  # Central configuration (all tunable params)
│   ├── ingestion/
│   │   ├── pubmed_fetcher.py      # PubMed/PMC corpus download
│   │   └── corpus_registry.py    # Paper metadata registry
│   ├── preprocessing/
│   │   ├── xml_parser.py          # JATS/PMC XML parser
│   │   ├── pdf_parser.py          # PyMuPDF PDF fallback parser
│   │   ├── cleaner.py             # Text cleaning
│   │   ├── chunker.py             # Section-aware sentence-boundary chunking
│   │   └── entity_extractor.py   # scispaCy + keyword NER
│   ├── retrieval/
│   │   ├── embedder.py            # Sentence-transformer encoder
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   ├── bm25_index.py          # BM25Okapi lexical index
│   │   ├── hybrid_retriever.py    # RRF hybrid fusion
│   │   └── reranker.py            # Cross-encoder reranker
│   ├── claims/
│   │   ├── query_normalizer.py    # Query concept extraction
│   │   └── claim_extractor.py    # Structured claim extraction
│   ├── nli/
│   │   └── nli_classifier.py     # Biomedical NLI
│   ├── context/
│   │   └── contradiction_analyzer.py  # Contextual conflict analysis
│   ├── evidence/
│   │   ├── relationship.py        # Evidence relationship aggregation
│   │   └── status.py             # SUPPORTED/REFUTED/INCONCLUSIVE
│   ├── generation/
│   │   ├── llm_client.py          # Multi-provider LLM wrapper
│   │   ├── prompt_builder.py      # Evidence-grounded prompt assembly
│   │   └── synthesizer.py         # Full pipeline orchestrator
│   └── evaluation/
│       ├── metrics.py             # Proxy evaluation metrics
│       └── baseline_runner.py    # DiaVeritas vs. baseline comparison
├── ui/
│   ├── app.py                     # Streamlit application
│   └── styles.css                 # Academic research-tool CSS
├── scripts/
│   ├── build_index.py             # Offline index builder
│   └── run_evaluation.py         # Evaluation runner
├── tests/
│   ├── test_chunker.py
│   ├── test_nli.py
│   └── test_evidence_status.py
├── data/
│   ├── raw/                       # Downloaded XML/PDF files
│   ├── processed/                 # chunks.jsonl + bm25_index.pkl
│   └── metadata/                  # corpus_registry.json
└── vector_store/                  # ChromaDB persistence directory
```

---

## Configuration

All tunable parameters are in `src/config.py` and mirrored in `.env`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | Biomedical SBERT |
| `NLI_MODEL` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | NLI classifier |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `CHUNK_SIZE_TOKENS` | `400` | Target chunk size (300–500 range) |
| `CHUNK_OVERLAP_PERCENT` | `15` | Chunk overlap (10–20% range) |
| `DENSE_RETRIEVAL_CANDIDATES` | `50` | Dense retrieval pool size |
| `BM25_CANDIDATES` | `50` | BM25 retrieval pool size |
| `RERANKER_TOP_K` | `10` | Final evidence items after reranking |

---

## Running Tests

```bash
py -m pytest tests/ -v
```

---

## Evaluation

```bash
py scripts/run_evaluation.py --output data/eval_results.json
```

---

## Known Limitations

- Retrieval may miss relevant papers not in the corpus
- PDF extraction may contain noise from tables and figures
- Claim extraction using LLM may occasionally be imprecise
- Biomedical NLI may misinterpret complex or nuanced claims
- Contextual comparison cannot detect all subtle clinical differences
- Evidence status depends on corpus coverage — not all T2D literature is indexed
- LLM can still produce errors in the synthesis step
- Confidence score is NOT clinically calibrated

---

## Team Structure

| Member | Responsibility |
|---|---|
| Member 1 | Corpus collection, XML/PDF parsing, cleaning, chunking |
| Member 2 | Embeddings, vector DB, BM25, hybrid retrieval, reranking |
| Member 3 | Claim extraction, NLI, contextual contradiction analysis |
| Member 4 | LLM integration, citations, UI, evaluation |

---

## Research Hypothesis

> "Explicit evidence analysis and contradiction-aware processing before generation will improve the evidence grounding and conflict awareness of RAG-based answers to Type 2 Diabetes treatment questions compared with standard RAG."

---

*This system is for research purposes only. It is not medical advice and is not validated for clinical use.*
