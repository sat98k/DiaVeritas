# DiaVeritas: Biomedical Evidence Synthesis & Contradiction Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTest Status](https://img.shields.io/badge/tests-47%20passed-brightgreen.svg)](tests/)
[![Dense Retrieval](https://img.shields.io/badge/retrieval-S--PubMedBert--MS--MARCO-orange.svg)](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO)
[![NLI Engine](https://img.shields.io/badge/NLI-DeBERTa--v3--base-purple.svg)](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Evidence synthesis & conflict resolution for Type 2 Diabetes clinical literature.**  
> *DiaVeritas retrieves, verifies, cross-examines, and synthesizes biomedical research to detect consensus and resolve clinical contradictions before generating citation-grounded answers.*

---

## 🔬 The Problem: Why Standard RAG Fails in Clinical NLP

Standard Retrieval-Augmented Generation (RAG) architectures are fundamentally unsuited for evidence-based medicine:

1. **Hallucinated Consensus:** When clinical trials present conflicting conclusions (e.g., intensive vs. standard glycemic control, or pharmacological therapy vs. intensive lifestyle modification), standard RAG naively concatenates chunks into the LLM context. The generative model tends to synthesize a polite, homogenized narrative, masking real scientific controversy.
2. **Whole-Chunk Noise Dilution in NLI:** Standard Natural Language Inference (NLI) classifiers fail when fed entire 400-word paper abstracts or sections. The critical outcome statement (e.g., *"Lifestyle intervention reduced incidence by 58% vs 31% for metformin"*) is drowned out by background study rationale, enrollment criteria, and safety monitoring details, causing the model to output **Neutral (90%+)**.
3. **Passive/Directional Query Distortion:** Clinical questions are frequently phrased passively or open-endedly (e.g., *"Does diabetes improve with exercise?"*). Comparing an open question directly against literature premises breaks directional NLI premise-hypothesis contracts.
4. **Lack of Evidence Calibration:** Standard systems output answers without reporting the underlying ratio of supporting vs. contradicting vs. inconclusive trial chunks.

---

## 💡 The DiaVeritas Solution

DiaVeritas replaces naive RAG with a structured multi-stage verification pipeline:

```
                          [ Clinical Question ]
                                    │
                                    ▼
                      [ Query Normalizer & Causal ]
                     [ Inverter (Hypothesis Form) ]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [ Dense Retrieval ]                   [ Sparse Retrieval ]
      (S-PubMedBert-MS-MARCO)                    (BM25Okapi)
                  └─────────────────┬─────────────────┘
                                    ▼
                      [ Reciprocal Rank Fusion (RRF) ]
                                    │
                                    ▼
                     [ Cross-Encoder Reranker ]
                   (ms-marco-MiniLM-L-6-v2) Top-K
                                    │
                                    ▼
                 [ Sentence-Salience NLI Preprocessing ]
               (Lexical Overlap + Clinical Outcome Scoring)
                                    │
                                    ▼
                     [ Biomedical DeBERTa-v3 NLI ]
                  (Entailment, Contradiction, Neutral)
                                    │
                                    ▼
               [ Contextual Contradiction Resolution ]
             (PICO/Design Discrepancy & Agreement Scoring)
                                    │
                                    ▼
               [ Calibrated Evidence Status Determination ]
                  (SUPPORTED / REFUTED / INCONCLUSIVE)
                                    │
                                    ▼
                  [ Evidence-Grounded LLM Synthesizer ]
               (Groq Llama-3.3-70B / Qwen-27B / GPT-4o)
                                    │
                                    ▼
             [ Structured Verdict & Interactive Clinical UI ]
```

### Core Methodological Innovations

1. **Sentence-Salience Premise Extraction (`src/nli/nli_classifier.py`):**
   - Automatically segments retrieved chunks into discrete sentences.
   - Scores candidate sentences using a composite clinical salience function:
     $$\text{Salience}(S) = \text{Overlap}(S, Q_{\text{content}}) + \omega_{\text{outcome}}\mathbb{I}(S \cap \mathcal{C}_{\text{outcome}}) - \omega_{\text{design}}\mathbb{I}(S \cap \mathcal{C}_{\text{protocol}})$$
     where $\mathcal{C}_{\text{outcome}}$ rewards clinical outcome cues (*reduced, increased, improved, hazard ratio, p < 0.05, incidence, prevented*) and $\mathcal{C}_{\text{protocol}}$ penalizes background study methodology.
   - Evaluates the top-3 salient sentences per chunk in batched DeBERTa-v3 inference, resolving the "10 Neutrals" failure mode without hardcoded query logic.

2. **Causal Inversion Query Normalization (`src/claims/query_normalizer.py`):**
   - Transforms passive and conversational inquiries (*"Does diabetes improve with exercise (like walking)?"*) into clean, directional causal hypotheses (*"Exercise improves diabetes."*).
   - Preserves specific clinical terms for lexical BM25 while passing natural clinical phrases to biomedical dense embedding encoders.

3. **Hybrid Dense-Sparse RRF Retrieval (`src/retrieval/hybrid_retriever.py`):**
   - Fuses `pritamdeka/S-PubMedBert-MS-MARCO` (semantic vector search) and `BM25Okapi` (exact keyword matching) using Reciprocal Rank Fusion ($k=60$):
     $$\text{RRF}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{60 + \text{Rank}_m(d)}$$
   - Followed by cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to prune the candidate pool to the top-10 most relevant evidence chunks.

4. **Curated Landmark Trial Ingestion (`src/ingestion/pubmed_fetcher.py`):**
   - Seeded with 18 landmark clinical trials across Type 2 Diabetes, lifestyle interventions, and cardiovascular outcomes, including:
     - **DPP (2002)** (PMID `11832527`) & **DPPOS** (PMIDs `19875686`, `26377189`)
     - **Finnish Diabetes Prevention Study** (PMID `11337921`)
     - **DARE Trial** (PMID `17876020`) & **HART-D Trial** (PMID `21098771`)
     - **Umpierre et al. Meta-analysis** (PMID `21540559`)
     - **EMPA-REG OUTCOME** (PMID `26378442`) & **DAPA-HF** (PMID `31535829`)
     - **UKPDS 34** (PMID `9742977`) & **LEADER** (PMID `27295427`)

5. **Consensus Calibration Engine (`src/evidence/status.py`):**
   - Objectively classifies research findings into:
     - **SUPPORTED:** Consistent entailment across literature chunks.
     - **REFUTED:** Majority contradiction across landmark trials.
     - **INCONCLUSIVE:** Bimodal distribution or insufficient decisive evidence.
   - Calculates a calibrated confidence score based on the margin of agreement and NLI prediction probabilities.

---

## 📊 Demonstrated Benchmark Results

| Clinical Research Question | Evidence Verdict | Calibration | Evidence Breakdown | Landmark Evidence Grounding |
|:---|:---:|:---:|:---:|:---|
| *"Is metformin more effective than lifestyle changes for prediabetes?"* | **REFUTED** | **High (91%)** | 5 Contradicting<br>0 Supporting<br>5 Neutral | Grounded in the **DPP Trial (PMID 11832527)**: Lifestyle intervention reduced diabetes incidence by 58% vs. 31% for metformin. |
| *"Does diabetes improve with exercise (like walking)?"* | **SUPPORTED** | **High (83%)** | 8 Supporting<br>0 Contradicting<br>2 Neutral | Grounded in the **DARE Trial (PMID 17876020)** and **Umpierre Meta-Analysis (PMID 21540559)**: Structured aerobic & resistance training significantly reduces HbA1c. |
| *"Does metformin reduce cardiovascular risk in T2D?"* | **INCONCLUSIVE** | **High (66%)** | 3 Supporting<br>1 Contradicting<br>6 Neutral | Correctly detects scientific debate between early **UKPDS 34** cardiovascular benefit claims and modern contemporary trials showing neutral CVD outcomes. |

---

## 📂 Repository Structure

```
DiaVeritas/
├── src/
│   ├── config.py                      # Central configuration & hyperparameters
│   ├── ingestion/
│   │   ├── pubmed_fetcher.py          # PubMed API fetcher & landmark trial ingest
│   │   └── corpus_registry.py        # Metadata registry for downloaded papers
│   ├── preprocessing/
│   │   ├── xml_parser.py              # JATS/PMC XML structured parser
│   │   ├── pdf_parser.py              # PyMuPDF fallback parser
│   │   ├── cleaner.py                 # Biomedical text cleaning & formatting
│   │   ├── chunker.py                 # Section-aware sentence-boundary chunker
│   │   └── entity_extractor.py       # scispaCy & clinical NER extraction
│   ├── retrieval/
│   │   ├── embedder.py                # S-PubMedBert sentence-transformer wrapper
│   │   ├── vector_store.py            # ChromaDB persistent vector database
│   │   ├── bm25_index.py              # BM25Okapi lexical index with tokenization
│   │   ├── hybrid_retriever.py        # Reciprocal Rank Fusion (RRF) pipeline
│   │   └── reranker.py                # MiniLM cross-encoder reranker
│   ├── claims/
│   │   ├── query_normalizer.py        # Causal hypothesis inversion & token cleaner
│   │   └── claim_extractor.py        # Structured PICO claim extractor
│   ├── nli/
│   │   └── nli_classifier.py         # Sentence-salience DeBERTa-v3 NLI classifier
│   ├── context/
│   │   └── contradiction_analyzer.py  # Contextual conflict & population discrepancy
│   ├── evidence/
│   │   ├── relationship.py            # Evidence relationship aggregation
│   │   └── status.py                 # Consensus determination & calibration
│   └── generation/
│       ├── llm_client.py              # Multi-provider LLM client (Groq, OpenAI, Gemini)
│       ├── prompt_builder.py          # Evidence-grounded citation prompt builder
│       └── synthesizer.py             # End-to-end pipeline orchestrator
├── ui/
│   ├── app.py                         # Streamlit clinical research dashboard
│   └── styles.css                     # Custom academic medical UI stylesheet
├── scripts/
│   ├── build_index.py                 # Offline corpus builder & indexer CLI
│   └── run_evaluation.py             # Baseline RAG vs DiaVeritas benchmark runner
├── tests/
│   ├── test_chunker.py                # Chunker boundary & metadata preservation tests
│   ├── test_nli.py                    # NLI classification & scoring tests
│   ├── test_query_normalizer.py        # Query inversion & normalization tests
│   └── test_evidence_status.py        # Consensus logic & calibration tests
├── data/                              # Data directories (raw XML, processed indices)
├── requirements.txt                   # Production Python dependencies
├── .env.example                       # Template for API keys & model settings
├── SETUP_GUIDE.md                     # In-depth setup and execution manual
└── README.md                          # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/dia-veritas/dia-veritas.git
cd dia-veritas

# Create and activate a virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional but recommended for biomedical NER):*
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz
```

### 3. Configure Environment Variables

Copy the example `.env` file and provide your LLM API key (Groq offers fast free-tier inference; OpenAI, Anthropic, or Google Gemini are also supported):

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` to set your credentials:
```ini
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional: NCBI credentials for faster PubMed downloads (avoids NCBI rate limits)
NCBI_API_KEY=
NCBI_EMAIL=your_email@domain.com
```

### 4. Build the Offline Index

Run the offline pipeline to download PubMed open-access XMLs and landmark trials, generate section-aware chunks, and compute dense embeddings and BM25 indices:

```bash
# Ingest 60+ clinical papers and landmark trials, chunk, embed, and index
python scripts/build_index.py --step all --max-papers 60 --fresh
```

### 5. Run the Streamlit Clinical Interface

Launch the interactive researcher dashboard:

```bash
streamlit run ui/app.py
```

Open your browser at `http://localhost:8501`. You can test preloaded clinical questions or type your own research queries.

---

## 🧪 Running the Test Suite

DiaVeritas includes comprehensive test coverage for chunking, query inversion, sentence-salience NLI, and evidence consensus logic:

```bash
pytest tests/ -v
```

All 47 unit tests should pass cleanly in under 1 second.

---

## 🔄 Deploying to Your GitHub Repository

If you are setting up or pushing to a new repository (`https://github.com/dia-veritas/dia-veritas`):

```bash
# 1. Verify your remote configuration
git remote -v

# 2. If needed, point origin to the new repo
git remote set-url origin https://github.com/dia-veritas/dia-veritas.git

# 3. Stage and commit your changes
git add .
git commit -m "feat: complete DiaVeritas biomedical evidence synthesis system"

# 4. Push to main branch
git push -u origin main
```

---

## ⚖️ Clinical & Academic Disclaimer

> **IMPORTANT:** DiaVeritas is an academic research prototype designed solely for natural language processing (NLP), evidence synthesis, and medical literature conflict resolution research.
>
> - It is **NOT** a medical device, clinical decision support system (CDSS), or diagnostic tool.
> - It does **NOT** provide medical advice, diagnosis, or treatment recommendations.
> - Information synthesized by this system must **NEVER** be used to guide patient care or clinical decisions.
> - Always consult qualified healthcare professionals and primary literature for medical inquiries.

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE). If you use DiaVeritas in your research, please cite:

```bibtex
@software{diaveritas2026,
  author = {DiaVeritas Research Team},
  title = {DiaVeritas: RAG-Based Biomedical Evidence Synthesis and Contradiction Resolution},
  year = {2026},
  url = {https://github.com/dia-veritas/dia-veritas}
}
```
