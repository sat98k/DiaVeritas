# DiaVeritas — Handover

**Repo:** https://github.com/sat98k/DiaVeritas.git
**Branch to use:** `testing2` (NOT `main` — see below)

This document is everything a teammate needs to pick up the project as-is:
where the latest code lives, how to configure it, one open decision that
needs team input, and how to verify everything still works before
submission.

---

## 1. Git Branch Alignment

> [!IMPORTANT]
> `origin/main` is currently sitting at an older commit (`7960c64`) and
> does **not** have the recent fixes or the current passing test suite.
> If you clone or pull `main`, you will be missing all of the recent
> defect fixes. You must check out `testing2`.

```bash
git fetch origin
git checkout testing2
git pull origin testing2
```

- [ ] Confirm you're on `testing2`, not `main`
- [ ] *(Optional, only if submission requires code to live on `main`)*
      merge `testing2` into `main` via PR once everything below is
      verified

**Latest commits on `testing2`:**
- `393c28a` — "Fix files updated" (`fixes`, `fixes.md` — 14-point error
  tracking log)
- `5aa2a76` — "Fixed the refute errors" (comparative_gate.py,
  contradiction_analyzer.py, claim_extractor.py, query_normalizer.py,
  status.py, synthesizer.py, entity_extractor.py, plus new tests
  test_comparative_gate.py and test_polarity_generalization.py)

---

## 2. Environment Configuration & API Keys

`.env` is gitignored (protects secrets) — copy `.env.example` to `.env`
and fill in:

**NCBI / PubMed Entrez:**
- Get an API key: [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/) → Generate API Key
- Without it, NCBI throttles to 3 requests/sec (slow, prone to HTTP 429
  bans on a 2,000-paper download). With it, 10 requests/sec.
```env
NCBI_API_KEY=your_actual_ncbi_api_key_here
NCBI_EMAIL=your_email@domain.com
```

**Groq API Key (clinical synthesis LLM):**
- Get a key: [Groq Console](https://console.groq.com/keys)
```env
GROQ_API_KEY=gsk_...
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
```

- [ ] `.env` created from `.env.example`
- [ ] NCBI key + email set
- [ ] Groq key set

---

## 3. Known Issue: PubMedBERT vs. MiniLM Embedding Mismatch (decision needed)

**Status:** Open — needs a team decision before final submission
**Severity:** Not a bug — a deliberate config choice worth a team call

### The exact situation

| | Documented default | Actual runtime |
|---|---|---|
| Model | `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical) | `all-MiniLM-L6-v2` (general-purpose) |
| Dimension | 768 | 384 |
| Size | ~420MB | ~80MB |

- `src/config.py`, `README.md`, `SETUP_GUIDE.md`, and `DiaVeritas_SRS.md`
  all state PubMedBERT as the model.
- `.env` currently overrides this to `all-MiniLM-L6-v2`.
- The current ChromaDB index (`vector_store/`, 15,513 chunks) was **built
  using MiniLM**, at 384 dimensions.

### Why it might matter for accuracy

A general-purpose embedding model is weaker at fine-grained biomedical
distinctions that a domain-tuned model handles better — e.g.
distinguishing "cardiovascular events" from "cardiovascular mortality,"
or connecting a drug-class term ("SGLT2 inhibitors") to its specific
member drugs at the embedding level. Some retrieval issues already
patched with manual synonym/class-expansion tables may be partly
compensating for this gap rather than the embedding model handling it
natively.

### ⚠️ The danger — read before touching `.env`

**Do not** just change `EMBEDDING_MODEL` to PubMedBERT in `.env` without
re-indexing. The two models don't share a vector space — ChromaDB will
throw a dimension mismatch error (`query dimension 768 != ChromaDB
dimension 384`) on every query if you do.

### Choose one path

- [ ] **Option A — Rebuild index with PubMedBERT (recommended for final
      submission, if time allows):**
  1. In `.env`, set:
     ```env
     EMBEDDING_MODEL=pritamdeka/S-PubMedBert-MS-MARCO
     ```
  2. Re-run index generation with `--fresh` so ChromaDB builds a clean
     768-dim collection.
  3. Verify retrieval works end-to-end with the new embeddings.
  4. Not a GPU/VRAM problem — PubMedBERT runs fine on CPU. It's a time
     cost (re-embedding 15,513 chunks), not new engineering work.

- [ ] **Option B — Keep `all-MiniLM-L6-v2` for lightweight CPU
      deployment:**
  1. Leave `.env` as-is.
  2. Update `README.md` and `SETUP_GUIDE.md` to explicitly document this
     as the active choice, e.g.:
     > "For low-latency CPU deployment, `all-MiniLM-L6-v2` (384-dim) is
     > configured as the active runtime embedder, while
     > `pritamdeka/S-PubMedBert-MS-MARCO` (768-dim) remains the
     > biomedical high-accuracy alternative, configurable via `.env`."

**This has not been decided yet — flagging for team input given
remaining time before the deadline.**

---

## 4. Downloading the Corpus (2,000+ papers)

`data/raw/`, `data/processed/`, and `vector_store/` are all gitignored —
a fresh clone has no local papers or vector index. Two options:

**A. Build from scratch:**
```bash
python scripts/build_2000_papers.py --target-papers 2000 --fresh
```
This automatically:
1. Downloads balanced clinical papers across 5 pillars (Nutrition,
   Pharmacotherapy, Exercise, Complications/CVD, Prediabetes) plus seeded
   landmark trials (UKPDS, ACCORD, ADVANCE, EMPA-REG, DAPA-HF, etc.)
2. Parses PMC full-text XML and PubMed abstracts
3. Chunks text (boundary-aware) into `data/processed/chunks.jsonl`
4. Runs clinical entity extraction (`enrich_chunks_with_entities`)
5. Computes embeddings with the configured `EMBEDDING_MODEL` and stores
   them in ChromaDB (`vector_store/`)
6. Builds the BM25 sparse index (`data/processed/bm25_index.pkl`)

**B. Transfer directly:** copy `data/` and `vector_store/` from an
existing local machine instead of re-downloading/re-indexing.

- [ ] Corpus present (built or copied)
- [ ] `vector_store/` present and matches the embedding model decided in
      Section 3

---

## 5. Verification Checklist (run before submitting)

**1. Full automated test suite:**
```bash
pytest tests/ -q
```
- [ ] All 104 tests pass (100%)
  - Covers `test_polarity_generalization.py` (17 tests), 
    `test_comparative_gate.py` (14 tests),
    `test_phase1_contextual_leakage.py`, plus all prior unit/integration
    tests.

**2. Streamlit UI sanity check:**
```bash
streamlit run ui/app.py
```
- [ ] UI starts cleanly, no import errors
- [ ] "Does metformin reduce cardiovascular mortality?" → **SUPPORTED**
      (~0.95 confidence)
- [ ] "Does metformin increase cardiovascular mortality?" → **REFUTED**
      (~0.90 confidence)
- [ ] "Which is better for lowering HbA1c: metformin or sulfonylureas?"
      → shows the Comparative Claim Verification warning (no false
      monotherapy-superiority claim)

**3. Benchmark run (EV-1 through EV-11):**
```bash
python scripts/run_clinical_benchmark.py
```
- [ ] Citation grounding warnings = 0
- [ ] Remaining benchmark mismatches (PHARM-001, EXER-006, EXER-010,
      COMP-008) are confirmed true corpus-coverage gaps (missing RCT
      trials in the fetched set), not pipeline logic errors

---

## Summary — what your teammate needs to do, in order

1. Check out `testing2` (not `main`)
2. Set up `.env` (NCBI + Groq keys)
3. **Decide Section 3** (PubMedBERT rebuild vs. keep MiniLM) — this
   affects step 4
4. Build or copy the corpus
5. Run the full verification checklist in Section 5
