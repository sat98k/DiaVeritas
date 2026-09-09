import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.generation.synthesizer import Synthesizer

bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
embedder = Embedder()
vector_store = VectorStore()
bm25 = BM25Index.load(bm25_path)
synth = Synthesizer(embedder, vector_store, bm25, llm_client=None)

question = "Does metformin reduce HbA1c levels in patients with Type 2 Diabetes?"
print("=== TRACING Q1 ===")
res = synth.run(question, mode="diaveritias")

print("\n--- TOP-10 RERANKED CHUNKS ---")
for idx, r in enumerate(res.reranked):
    cid = r.get("chunk_id")
    pid = r.get("paper_id")
    score = r.get("reranker_score")
    title = r.get("title") or "Unknown Title"
    text = r.get("text", "")[:200].replace("\n", " ")
    print(f"[{idx+1}] chunk_id={cid} | paper_id={pid} | score={score:.3f} | title={title[:40]}")
    print(f"     text: {text}...\n")

print("--- EVIDENCE SUMMARY ---")
ev_dict = res.evidence_summary_dict
sup = ev_dict.get("supporting", [])
con = ev_dict.get("contradicting", [])
ctx = ev_dict.get("contextual", [])
neu = ev_dict.get("neutral", [])

print(f"Supporting count: {len(sup)}")
for idx, s in enumerate(sup):
    print(f"  Sup [{idx+1}]: chunk_id={s.get('chunk_id')} | paper_id={s.get('paper_id')} | title={s.get('title')}")
    print(f"        text: {s.get('text')[:150]}")

print(f"\nContradicting count: {len(con)}")
for idx, c in enumerate(con):
    print(f"  Con [{idx+1}]: chunk_id={c.get('chunk_id')} | paper_id={c.get('paper_id')} | text={c.get('text')[:150]}")

print(f"\nContextual count: {len(ctx)}")
for idx, c in enumerate(ctx):
    print(f"  Ctx [{idx+1}]: chunk_id={c.get('chunk_id')} | paper_id={c.get('paper_id')} | text={c.get('text')[:150]}")

print(f"\nNeutral count: {len(neu)}")

distinct_pids = {s.get("paper_id") for s in sup if s.get("paper_id")}
print(f"\nDistinct supporting paper IDs from dicts ({len(distinct_pids)}): {distinct_pids}")

s_dict = res.status_result_dict
print("\n--- FINAL VERDICT ---")
print("Status:", res.status)
print("Confidence Breakdown:", s_dict.get("confidence_breakdown"))
print("Rationale:", s_dict.get("rationale"))
