# DiaVeritas: Comprehensive Setup & Developer Guide

This guide provides exhaustive, end-to-end instructions for installing, configuring, indexing, testing, and deploying **DiaVeritas** on Windows, Linux, and macOS.

---

## 📋 Table of Contents
1. [System Requirements](#1-system-requirements)
2. [Environment Setup & Installation](#2-environment-setup--installation)
3. [Configuration & Environment Variables (`.env`)](#3-configuration--environment-variables-env)
4. [Offline Corpus Building & Indexing](#4-offline-corpus-building--indexing)
5. [Running the Interactive Web Application](#5-running-the-interactive-web-application)
6. [Testing & Evaluation Suite](#6-testing--evaluation-suite)
7. [Architecture & Pipeline Reference](#7-architecture--pipeline-reference)
8. [Troubleshooting & FAQs](#8-troubleshooting--faqs)
9. [Pushing to a New GitHub Repository](#9-pushing-to-a-new-github-repository)

---

## 1. System Requirements

- **Operating System:** Windows 10/11, macOS 12+, or Ubuntu 20.04+ (or any modern Linux distribution)
- **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14
- **Hardware:**
  - **RAM:** Minimum 8 GB (16 GB recommended for concurrent model inference)
  - **Storage:** ~3 GB free disk space (for local transformer weights and ChromaDB vector cache)
  - **Inference:** Optimized for standard CPU inference; CUDA/MPS GPU will be automatically utilized if available.

---

## 2. Environment Setup & Installation

### Step 2.1: Clone or Navigate to the Repository

```bash
# If cloning fresh from GitHub:
git clone https://github.com/dia-veritas/dia-veritas.git
cd dia-veritas
```

### Step 2.2: Create and Activate a Clean Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> **Note:** If PowerShell displays an execution policy error (`script execution is disabled on this system`), run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> .\venv\Scripts\Activate.ps1
> ```

#### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 2.3: Upgrade Package Managers & Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 2.4: (Optional) Install Biomedical NER Models
For enhanced clinical entity recognition during document preprocessing:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz
```

---

## 3. Configuration & Environment Variables (`.env`)

DiaVeritas ships with an annotated configuration template in `.env.example`.

### Step 3.1: Initialize `.env`

```bash
# Windows (PowerShell / CMD)
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

### Step 3.2: Configure Your LLM Provider

Open `.env` in any text editor. DiaVeritas is designed to work with any major provider:

#### Option A: Groq (Recommended — Ultra-Fast & Free Tier Available)
1. Sign up at [console.groq.com](https://console.groq.com/) and generate an API key.
2. Update `.env`:
   ```ini
   LLM_PROVIDER=groq
   LLM_MODEL=llama-3.3-70b-versatile
   GROQ_API_KEY=gsk_your_actual_groq_api_key
   ```
   *(Alternative models on Groq: `qwen-2.5-32b`, `llama3-70b-8192`)*

#### Option B: OpenAI
```ini
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-your_openai_api_key
```

#### Option C: Google Gemini
```ini
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
GOOGLE_API_KEY=your_gemini_api_key
```

#### Option D: Anthropic Claude
```ini
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Step 3.3: (Optional) PubMed / NCBI Configuration
To download papers faster without NCBI rate-limiting (allows up to 10 requests/sec instead of 3):
```ini
NCBI_API_KEY=your_ncbi_api_key
NCBI_EMAIL=your_email@university.edu
```

---

## 4. Offline Corpus Building & Indexing

Before querying the system, you need an indexed corpus of clinical literature. DiaVeritas features an automated offline pipeline that searches PubMed, downloads open-access PMC XMLs and landmark clinical trials, segments sections into boundary-aware chunks, and builds both ChromaDB vector embeddings and BM25 inverted indices.

### Build the Full Benchmark Corpus
```bash
python scripts/build_index.py --step all --max-papers 60 --fresh
```

#### What this command does:
1. **Ingestion (`--step ingest`):**
   - Queries PubMed for recent Type 2 Diabetes literature across medications, lifestyle, diet, and outcomes.
   - Automatically injects **18 landmark clinical trials** (DPP, DPPOS, Finnish DPS, HART-D, DARE, Umpierre meta-analysis, EMPA-REG, DAPA-HF, UKPDS, LEADER).
   - Parses structured JATS XML files into sections (Introduction, Methods, Results, Discussion).
   - Cleans text and splits into 400-token chunks with 15% overlap at sentence boundaries.
   - Writes `data/processed/chunks.jsonl`.
2. **Indexing (`--step index`):**
   - Computes dense biomedical embeddings using `pritamdeka/S-PubMedBert-MS-MARCO` and stores them in ChromaDB (`vector_store/`).
   - Builds a lexical inverted index with `rank-bm25` saved to `data/processed/bm25_index.pkl`.

> **Tip:** If you only want to re-index existing chunks without re-downloading PubMed XMLs:
> ```bash
> python scripts/build_index.py --step index --fresh
> ```

---

## 5. Running the Interactive Web Application

Launch the Streamlit clinical research dashboard:

```bash
streamlit run ui/app.py
```

The application will launch and be accessible at:
```
http://localhost:8501
```

### Dashboard Features:
- **Sample Clinical Questions:** Click any pre-curated query (Metformin vs. Lifestyle in Prediabetes, Exercise in T2D, GLP-1 CKD Safety, etc.) or type any clinical hypothesis.
- **Evidence Verdict Banner:** Displays **SUPPORTED** (Green), **REFUTED** (Red), or **INCONCLUSIVE** (Amber).
- **Confidence Calibration Meter:** Displays statistical agreement percentage across retrieved literature.
- **3-Way Evidence Breakdown:** Exact count of Supporting, Contradicting, Contextual, and Neutral chunks.
- **Clinical Evidence Synthesis:** Direct structured answer, evidence breakdown table, conflict analysis, and clinical caveats.
- **Pipeline Reasoning Trail:** Full transparent audit trail showing normalized causal hypotheses, dense vs. BM25 ranks, extracted salient premise sentences, and DeBERTa-v3 probabilities.
- **Literature Bibliography:** Direct links to PubMed and PMIDs for all cited literature chunks.

---

## 6. Testing & Evaluation Suite

### 6.1: Unit Tests
Run the comprehensive 47-test test suite:
```bash
pytest tests/ -v
```
**Coverage includes:**
- Section-aware chunking and token preservation (`test_chunker.py`)
- Causal query inversion and parenthetical cleaning (`test_query_normalizer.py`)
- Sentence-salience NLI premise scoring and DeBERTa inference (`test_nli.py`)
- Evidence consensus determination and confidence calibration (`test_evidence_status.py`)

### 6.2: Benchmark Evaluation Runner
Compare DiaVeritas against naive standard RAG across benchmark queries:
```bash
python scripts/run_evaluation.py --output data/eval_results.json
```

---

## 7. Architecture & Pipeline Reference

```
Question: "Is metformin more effective than lifestyle changes for prediabetes?"
  │
  ├─ 1. Query Normalizer (Hypothesis: "Metformin is more effective than lifestyle changes for prediabetes.")
  │
  ├─ 2. Hybrid Retrieval:
  │      - S-PubMedBert MS-MARCO (Dense semantic search, top-50)
  │      - BM25Okapi (Sparse keyword search, top-50)
  │      - Reciprocal Rank Fusion (RRF, k=60)
  │
  ├─ 3. Cross-Encoder Reranking:
  │      - ms-marco-MiniLM-L-6-v2 prunes pool to top-10 chunks
  │
  ├─ 4. Sentence-Salience NLI Premise Extraction:
  │      - Evaluates token overlap + clinical outcome cues (reduced, hazard ratio, p<0.05)
  │      - Filters out background protocol definitions
  │      - Selects top candidate sentences
  │
  ├─ 5. Biomedical DeBERTa-v3 NLI:
  │      - Batched inference yields: 5 Contradictions, 0 Entailments, 5 Neutrals
  │
  ├─ 6. Consensus Engine:
  │      - Status: REFUTED
  │      - Confidence: 91% (High)
  │
  └─ 7. LLM Evidence Synthesizer:
         - Generates structured, citation-grounded answer grounded in DPP 2002 trial (PMID 11832527).
```

---

## 8. Troubleshooting & FAQs

### Q: Why do I get "Address already in use" when running Streamlit?
**A:** Another instance of Streamlit is already running on port 8501. Either kill the process or launch on an alternative port:
```bash
streamlit run ui/app.py --server.port 8502
```

### Q: What if I don't have a GPU?
**A:** DiaVeritas is fully optimized for CPU execution. The embedding model (`S-PubMedBert`), reranker (`MiniLM`), and NLI classifier (`DeBERTa-v3`) run comfortably in RAM with inference times of ~1-2 seconds per query on standard CPUs.

### Q: Why do some queries return "INCONCLUSIVE"?
**A:** `INCONCLUSIVE` is a first-class, intentional verdict in clinical evidence synthesis. It occurs when:
1. Genuine medical controversy exists (e.g., conflicting findings across trials).
2. The retrieved chunks have insufficient decisive outcome evidence.
Unlike commercial chatbots that hallucinate an answer, DiaVeritas preserves scientific fidelity.

---

## 9. Pushing to a New GitHub Repository

To push this complete codebase to your target repository (`https://github.com/dia-veritas/dia-veritas`):

### Step 9.1: Verify Remote URL
```bash
git remote -v
```
If `origin` is not pointing to your target repository, set it:
```bash
git remote set-url origin https://github.com/dia-veritas/dia-veritas.git
```

### Step 9.2: Stage and Commit All Changes
```bash
git add .
git status
git commit -m "feat: complete DiaVeritas biomedical evidence synthesis system"
```

### Step 9.3: Push to GitHub

#### Using HTTPS (Personal Access Token):
```bash
git push -u origin main
```
When prompted for authentication:
- **Username:** Your GitHub username
- **Password:** Your GitHub Personal Access Token (PAT) with `repo` scope (generated at GitHub Settings -> Developer Settings -> Personal Access Tokens).

#### Using SSH (if you have SSH keys configured):
```bash
git remote set-url origin git@github.com:dia-veritas/dia-veritas.git
git push -u origin main
```

Your new repository is now fully configured, documented, and ready for use!
