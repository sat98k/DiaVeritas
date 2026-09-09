import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.generation.synthesizer import Synthesizer, _find_cited_chunks, _passage_supports_sentence, enforce_strict_grounding
from src.generation.llm_client import LLMClient
from src.evidence.status import _distinct_study_ids

print("Initializing models...")
bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
embedder = Embedder()
vector_store = VectorStore()
bm25 = BM25Index.load(bm25_path)

llm = LLMClient()
llm._load()

synth_llm = Synthesizer(embedder, vector_store, bm25, llm_client=llm)
synth_base = Synthesizer(embedder, vector_store, bm25, llm_client=None)

print("\n" + "=" * 80)
print("TEST GROUP 2 — LIVE Q1 RE-RUN & DEDUPLICATION TRACE")
print("=" * 80)

q1 = "Does metformin reduce HbA1c levels in patients with Type 2 Diabetes?"
res_q1 = synth_base.run(q1, mode="diaveritias")

print(f"Query: '{q1}'")
print(f"Final Verdict: {res_q1.status}")
s_dict1 = res_q1.status_result_dict
print(f"Confidence: {s_dict1.get('confidence_label')} ({s_dict1.get('confidence_score')})")
print(f"Evidence counts: Supporting={s_dict1.get('n_supporting')}, Contradicting={s_dict1.get('n_contradicting')}, Contextual={s_dict1.get('n_contextual')}, Neutral={s_dict1.get('n_neutral')}")
print(f"Weights: W_sup={s_dict1.get('weighted_support')}, W_con={s_dict1.get('weighted_contradiction')}, W_ctx={s_dict1.get('contextual_weight')}")
print(f"Rationale: {s_dict1.get('rationale')}")

print("\n--- TOP-10 RERANKED PASSAGES ---")
for idx, r in enumerate(res_q1.reranked):
    cid = r.get("chunk_id")
    pid = r.get("paper_id")
    score = r.get("reranker_score", 0.0)
    title = r.get("title") or "Unknown Title"
    text = r.get("text", "")[:180].replace("\n", " ")
    print(f"[{idx+1}] chunk_id={cid} | paper_id={pid} | score={score:.3f}")
    print(f"     Title: {title[:70]}")
    print(f"     Text: {text}...\n")

print("--- SUPPORTING PASSAGES & DEDUPLICATION ---")
sup_items = res_q1.evidence_summary_dict.get("supporting", [])
print(f"Total supporting passages: {len(sup_items)}")
for idx, it in enumerate(sup_items, 1):
    print(f"  [{idx}] chunk_id={it.get('chunk_id')} | paper_id={it.get('paper_id')} | title={it.get('title')[:60]}")

# Distinct IDs extracted
# In status.py, _distinct_study_ids is called on evidence items. Let's inspect paper_ids:
extracted_ids = [it.get('paper_id') or it.get('chunk_id') for it in sup_items]
distinct_set = set(extracted_ids)
print(f"Extracted IDs list: {extracted_ids}")
print(f"Distinct study IDs set: {distinct_set} (count: {len(distinct_set)})")

print("\n" + "=" * 80)
print("TEST GROUP 4.2 — LIVE LLM-BACKED GENERATION & STRICT GROUNDING TRACE")
print("=" * 80)

q2 = "Do SGLT2 inhibitors reduce the risk of hospitalization for heart failure in patients with Type 2 Diabetes?"
print(f"Running LLM-backed query: '{q2}'")
res_q2_llm = synth_llm.run(q2, mode="diaveritias")

print(f"\nVerdict: {res_q2_llm.status}")
print(f"Confidence: {res_q2_llm.status_result_dict.get('confidence_label')} ({res_q2_llm.status_result_dict.get('confidence_score')})")
print(f"Ungrounded claims stripped count: {len(res_q2_llm.stripped_claims)}")
for idx, sc in enumerate(res_q2_llm.stripped_claims, 1):
    print(f"  Stripped [{idx}]: {sc}")

print("\n--- GENERATED ANSWER WITH CITATIONS ---")
print(res_q2_llm.answer)

print("\n--- SENTENCE-LEVEL CITATION ATTRIBUTION & GROUNDING CHECK ---")
paragraphs = [p for p in res_q2_llm.answer.split("\n\n") if p.strip()]
import re
for p_idx, p in enumerate(paragraphs, 1):
    if p.strip().startswith(("#", "|", "*This analysis", "**Evidence Status")):
        continue
    sentences = re.split(r"(?<=[.!?])\s+", p)
    for s_idx, s in enumerate(sentences, 1):
        s_clean = s.strip()
        if len(s_clean) < 25:
            continue
        c_matches = re.findall(r"\[.+?\]", s_clean)
        if c_matches:
            print(f"Sentence [{p_idx}.{s_idx}]: {s_clean}")
            print(f"   Citation markers: {c_matches}")
            # Check support in evidence
            s_words = set(re.findall(r"\b\w{4,}\b", s_clean.lower())) - {"frontiers", "endocrinology", "cureus", "unknown", "reported"}
            matches = [r for r in res_q2_llm.reranked if len(s_words & set(re.findall(r"\b\w{4,}\b", r.get("text", "").lower()))) >= 3]
            print(f"   Supporting evidence passages in top-10: {len(matches)} matches")
            print()

print("\n" + "=" * 80)
print("TEST GROUP 5.2 — CONSOLIDATED MANUAL TEST SET LIVE RUN")
print("=" * 80)

manual_tests = [
    ("M1_Coffee_Sugar", "Does avoiding added sugar in coffee prevent progression from prediabetes to Type 2 Diabetes?"),
    ("M2_Rosiglitazone", "Does rosiglitazone increase the risk of myocardial infarction in patients with Type 2 Diabetes?"),
    ("M3_SGLT2_HF", "Do SGLT2 inhibitors reduce the risk of hospitalization for heart failure in patients with Type 2 Diabetes?"),
    ("M4_Forward_Metformin", "Does metformin reduce HbA1c levels in patients with Type 2 Diabetes?"),
    ("M5_Reversed_Metformin", "Does metformin increase the risk of cardiovascular mortality in patients with Type 2 Diabetes?"),
]

for mid, mq in manual_tests:
    print(f"\n--- Running [{mid}]: '{mq}' ---")
    m_res = synth_base.run(mq, mode="diaveritias")
    ms_dict = m_res.status_result_dict
    m_status = m_res.status
    m_conf = f"{ms_dict.get('confidence_label')} ({ms_dict.get('confidence_score')})"
    m_sup = ms_dict.get("n_supporting")
    m_con = ms_dict.get("n_contradicting")
    m_ctx = ms_dict.get("n_contextual")
    m_neu = ms_dict.get("n_neutral")
    m_wsup = ms_dict.get("weighted_support")
    m_wcon = ms_dict.get("weighted_contradiction")
    m_wctx = ms_dict.get("contextual_weight")
    
    print(f"Verdict: {m_status} | Confidence: {m_conf}")
    print(f"Breakdown: Sup={m_sup}, Con={m_con}, Ctx={m_ctx}, Neu={m_neu}")
    print(f"Weights: W_sup={m_wsup}, W_con={m_wcon}, W_ctx={m_wctx}")
