# Building the 2000-Paper Multi-Domain Clinical Dataset

This document serves as the direct execution specification for building or rebuilding DiaVeritas's comprehensive, multi-pillar clinical literature index (1,000 to 2,000 papers).

---

## 🤖 Prompt for Future AI Agents

If you are an AI assistant tasked with rebuilding or expanding the dataset, simply run:

```bash
python scripts/build_2000_papers.py --target-papers 2000 --fresh
```
*(Or `--target-papers 1000` for a faster ~8-minute build).*

---

## 🔬 Clinical Domain Composition

Rather than downloading generic molecular biology papers, the dataset builder partitions literature across **5 core clinical pillars**:

1. **Nutrition & Dietary Epidemiology (400 papers):**
   - Coffee and caffeine consumption vs. T2D risk (Ding 2014, Carlström 2018, Jiang 2014).
   - Sugar-sweetened beverages & metabolic syndrome (Malik 2010).
   - Mediterranean diet & cardiovascular prevention (PREDIMED trial, Estruch 2018).
   - Low-carbohydrate, ketogenic, intermittent fasting, and dietary fiber trials.

2. **Pharmacotherapy (400 papers):**
   - First-line biguanides (Metformin, UKPDS 34).
   - SGLT2 inhibitors (Empagliflozin EMPA-REG, Dapagliflozin DAPA-HF/DECLARE-TIMI 58, Canagliflozin CREDENCE).
   - GLP-1 receptor agonists & dual co-agonists (Semaglutide SUSTAIN-6/STEP-1, Tirzepatide SURPASS-2, Liraglutide LEADER).
   - Sulfonylureas, DPP-4 inhibitors, and basal/bolus insulin regimens.

3. **Exercise & Physical Activity (400 papers):**
   - Structured aerobic vs. resistance vs. combined training (DARE trial, HART-D trial).
   - Dose-response exercise meta-analyses (Umpierre 2011).
   - High-intensity interval training (HIIT), walking, and sedentary behavior interruption.

4. **Cardiovascular, Renal & Microvascular Complications (400 papers):**
   - Major Adverse Cardiovascular Events (3-point MACE, cardiovascular mortality).
   - Diabetic kidney disease (DKD/CKD progression, albuminuria reduction, eGFR slopes).
   - Diabetic retinopathy and peripheral neuropathy trials.

5. **Prediabetes Prevention & Diabetes Remission (400 papers):**
   - Intensive lifestyle intervention vs. metformin in prediabetes (DPP 2002, DPPOS 10/15-year follow-up, Finnish DPS).
   - Primary care-led dietary weight management for diabetes remission (DiRECT trial).
   - Impaired fasting glucose (IFG) and impaired glucose tolerance (IGT) progression.

---

## ⚡ Quick Execution Commands

### Full 2,000 Paper Build (Comprehensive Production Corpus)
```bash
python scripts/build_2000_papers.py --target-papers 2000 --fresh
```
- **Expected Time:** ~15–18 minutes (on CPU)
- **Expected Yield:** ~2,000 papers $\to$ ~20,000–25,000 section-aware chunks
- **Disk Usage:** ~120 MB (ChromaDB + BM25)

### Fast 1,000 Paper Build (Balanced Benchmark Corpus)
```bash
python scripts/build_2000_papers.py --target-papers 1000 --fresh
```
- **Expected Time:** ~8–10 minutes (on CPU)
- **Expected Yield:** ~1,000 papers $\to$ ~10,000–12,000 chunks
- **Disk Usage:** ~60 MB

### Using the Standard Index Script (Alternative)
The standard build script also automatically uses domain diversification when `max-papers >= 100`:
```bash
python scripts/build_index.py --step all --max-papers 2000 --fresh
```

---

## 🛠️ Performance & Scalability Guarantees

- **No Online Slowdown:** Because hybrid retrieval (ChromaDB vector cosine similarity + BM25Okapi) operates in logarithmic time, searching 25,000 chunks takes **<15 milliseconds**. Downstream Cross-Encoder reranking and DeBERTa NLI inference only evaluate the **top 10** chunks, keeping query response times constant at **~2 to 3 seconds**.
- **Rate-Limit Respect:** Biopython Entrez respects NCBI rate limits (3 requests/sec default; 10 requests/sec if `NCBI_API_KEY` is specified in `.env`).

---

## ✅ Verification

After building the dataset, verify its integrity:
```bash
# Check chunk count
python -c "import json; print(len(open('data/processed/chunks.jsonl', encoding='utf-8').readlines()), 'chunks loaded')"

# Run test suite
pytest tests/ -v

# Launch the dashboard
streamlit run ui/app.py
```
