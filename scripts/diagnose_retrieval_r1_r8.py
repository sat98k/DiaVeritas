# =============================================================================
# scripts/diagnose_retrieval_r1_r8.py
#
# Phase 3 Step 3a: Mandatory Diagnostic Script for R1-R8 query battery.
# Analyzes BM25-only, Dense-only, RRF-fused, and Reranked hits to isolate
# where class-level queries (SGLT2 inhibitors) drop specific drug evidence.
# =============================================================================

from __future__ import annotations

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.config import settings
from src.claims.query_normalizer import normalize_query
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker

R_QUERIES = [
    ("R1", "Do SGLT2 inhibitors reduce heart failure hospitalization in patients with Type 2 Diabetes?"),
    ("R2", "Do SGLT-2 inhibitors reduce hospitalization for heart failure in T2DM?"),
    ("R3", "Do sodium-glucose cotransporter-2 inhibitors reduce hospitalization for heart failure in Type 2 Diabetes?"),
    ("R4", "Does SGLT2 inhibitor therapy reduce HHF in Type 2 Diabetes?"),
    ("R5", "Do empagliflozin and dapagliflozin reduce heart failure hospitalization in Type 2 Diabetes?"),
    ("R6", "What is the effect of SGLT2 inhibitors on heart failure hospitalization in Type 2 Diabetes?"),
    ("R7", "What cardiovascular benefits do SGLT2 inhibitors provide in Type 2 Diabetes?"),
    ("R8", "Do SGLT2 inhibitors reduce the risk of heart failure hospitalization compared with placebo in patients with Type 2 Diabetes?"),
]

TARGET_KEYWORDS = ["empagliflozin", "dapagliflozin", "canagliflozin", "empa-reg", "dapa-hf", "declare"]


def run_diagnostic():
    print("=" * 80)
    print("STEP 3A DIAGNOSTIC: R1 - R8 RETRIEVAL TRACE")
    print("=" * 80)

    embedder = Embedder()
    vector_store = VectorStore()
    bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
    bm25 = BM25Index.load(bm25_path)
    hybrid = HybridRetriever(embedder, vector_store, bm25)
    reranker = CrossEncoderReranker()

    results_summary = {}

    for q_id, q_text in R_QUERIES:
        print(f"\n[{q_id}] Query: \"{q_text}\"")
        norm = normalize_query(q_text)
        search_query = norm["normalized_text"]
        dense_query = norm.get("dense_query", q_text)

        # 1. Dense retrieval (top 15)
        q_vec = embedder.encode_query(dense_query)
        dense_hits = vector_store.query(q_vec, n_results=15)

        # 2. BM25 retrieval (top 15)
        bm25_hits = bm25.query(search_query, n_results=15)

        # 3. RRF Fusion (top 15)
        for h in dense_hits:
            h["dense_score"] = h.get("score", 0.0)
            h["bm25_score"] = 0.0
        for h in bm25_hits:
            h["bm25_score"] = h.get("score", 0.0)
            h["dense_score"] = 0.0

        rrf_hits = reciprocal_rank_fusion([dense_hits, bm25_hits], k=60)[:15]

        # 4. Post-Reranking (top 10 from 25 hybrid candidates)
        candidates_for_rerank = hybrid.retrieve(q_text, dense_query=dense_query, bm25_query=search_query)
        reranked_hits = reranker.rerank(q_text, candidates_for_rerank, top_k=10)

        def count_targets(hits):
            found = []
            for idx, h in enumerate(hits, 1):
                text = (h.get("text", "") or "").lower()
                chunk_obj = h.get("chunk")
                title = (getattr(chunk_obj, "title", "") or "").lower() if chunk_obj else ""
                matched = [kw for kw in TARGET_KEYWORDS if kw in text or kw in title]
                if matched:
                    found.append((idx, h.get("chunk_id", "?"), matched))
            return found

        dense_targets = count_targets(dense_hits)
        bm25_targets = count_targets(bm25_hits)
        rrf_targets = count_targets(rrf_hits)
        rerank_targets = count_targets(reranked_hits)

        print(f"  BM25 matches ({len(bm25_targets)}/15): {[f'#{r[0]} ({r[2]})' for r in bm25_targets]}")
        print(f"  Dense matches ({len(dense_targets)}/15): {[f'#{r[0]} ({r[2]})' for r in dense_targets]}")
        print(f"  RRF matches   ({len(rrf_targets)}/15): {[f'#{r[0]} ({r[2]})' for r in rrf_targets]}")
        print(f"  Reranked top10({len(rerank_targets)}/10): {[f'#{r[0]} ({r[2]})' for r in rerank_targets]}")

        results_summary[q_id] = {
            "query": q_text,
            "bm25_targets": len(bm25_targets),
            "dense_targets": len(dense_targets),
            "rrf_targets": len(rrf_targets),
            "rerank_targets": len(rerank_targets),
        }

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY:")
    print("=" * 80)
    for q_id, stats in results_summary.items():
        print(f"{q_id}: BM25={stats['bm25_targets']} | Dense={stats['dense_targets']} | RRF={stats['rrf_targets']} | Rerank={stats['rerank_targets']}")


if __name__ == "__main__":
    run_diagnostic()
