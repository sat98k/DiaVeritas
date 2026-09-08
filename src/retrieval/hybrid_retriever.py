# =============================================================================
# src/retrieval/hybrid_retriever.py
#
# Reciprocal Rank Fusion (RRF) hybrid retrieval.
#
# Combines dense (vector) and BM25 rankings into a single ranked list.
#
# Why RRF?
#   - Simple, transparent, parameter-light (only one constant: k)
#   - Consistently strong empirically across retrieval tasks
#   - Each list contributes based on rank position, not raw score magnitude
#     (scores from dense and BM25 are not directly comparable)
#
# RRF formula:
#   RRF_score(d) = Σ 1 / (k + rank_i(d))
#   where rank_i(d) is the rank of document d in list i (1-indexed)
#   and k is a constant (default 60, following Cormack et al. 2009)
#
# Reference: Cormack, Clarke, & Buettcher (2009) — Reciprocal Rank Fusion.
# =============================================================================

from __future__ import annotations

from typing import List, Dict, Any, Optional

from loguru import logger

from src.config import settings
from src.preprocessing.chunker import Chunk
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of result lists. Each result must have "chunk_id".
                      Results should already be sorted best-first.
        k: RRF constant. Higher k reduces the impact of top ranks.
           Default 60 follows the original paper.

    Returns:
        Single fused and re-ranked list, sorted by RRF score descending.
        Each item has all fields from the original result dicts plus:
        - "rrf_score": float
        - "source_ranks": dict mapping source name to rank
    """
    rrf_scores: Dict[str, float] = {}
    chunk_data: Dict[str, Dict[str, Any]] = {}  # chunk_id → result dict
    source_ranks: Dict[str, Dict[str, int]] = {}  # chunk_id → {source: rank}

    for list_idx, result_list in enumerate(ranked_lists):
        source_name = f"list_{list_idx}"
        for rank_0based, result in enumerate(result_list):
            cid = result["chunk_id"]
            rank_1based = rank_0based + 1
            rrf_contribution = 1.0 / (k + rank_1based)

            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            if cid not in chunk_data:
                chunk_data[cid] = result.copy()
                source_ranks[cid] = {}
            source_ranks[cid][source_name] = rank_1based

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused = []
    for final_rank, cid in enumerate(sorted_ids):
        item = chunk_data[cid].copy()
        item["rrf_score"] = rrf_scores[cid]
        item["source_ranks"] = source_ranks[cid]
        item["rank"] = final_rank + 1
        fused.append(item)

    return fused


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Runs dense + BM25 retrieval and fuses results with RRF.

    This is the main retrieval component used by the online pipeline.

    Usage:
        retriever = HybridRetriever(embedder, vector_store, bm25_index)
        candidates = retriever.retrieve("Does metformin reduce CV risk?")
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: Optional[int] = None,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._rrf_k = rrf_k or settings.rrf_k

    def retrieve(
        self,
        query: str,
        dense_candidates: Optional[int] = None,
        bm25_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate evidence chunks using hybrid RRF fusion.

        Args:
            query: User query string (raw or normalized).
            dense_candidates: Number of dense retrieval candidates.
                              Defaults to settings.dense_retrieval_candidates.
            bm25_candidates: Number of BM25 candidates.
                              Defaults to settings.bm25_candidates.

        Returns:
            RRF-fused ranked list of candidate chunks, each with:
            - chunk_id, text, chunk (Chunk object)
            - dense_score, bm25_score (original scores from each retriever)
            - rrf_score (fused score)
            - source_ranks (rank in each individual list)
        """
        n_dense = dense_candidates or settings.dense_retrieval_candidates
        n_bm25 = bm25_candidates or settings.bm25_candidates

        # --- Dense retrieval ---
        logger.info(f"Dense retrieval: query='{query[:60]}...', n={n_dense}")
        query_vec = self._embedder.encode_query(query)
        dense_results = self._vector_store.query(query_vec, n_results=n_dense)

        # Tag scores for later transparency
        for item in dense_results:
            item["dense_score"] = item.get("score", 0.0)
            item["bm25_score"] = 0.0

        # --- BM25 retrieval ---
        logger.info(f"BM25 retrieval: n={n_bm25}")
        bm25_results = self._bm25.query(query, n_results=n_bm25)

        for item in bm25_results:
            item["bm25_score"] = item.get("score", 0.0)
            item["dense_score"] = 0.0

        # --- RRF fusion ---
        logger.info("Fusing with RRF...")
        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self._rrf_k,
        )

        # Merge scores: if a chunk appears in both lists, preserve both scores
        dense_score_map = {r["chunk_id"]: r["dense_score"] for r in dense_results}
        bm25_score_map = {r["chunk_id"]: r["bm25_score"] for r in bm25_results}

        for item in fused:
            cid = item["chunk_id"]
            item["dense_score"] = dense_score_map.get(cid, 0.0)
            item["bm25_score"] = bm25_score_map.get(cid, 0.0)

        logger.info(f"Hybrid retrieval: {len(fused)} unique candidates after RRF.")
        return fused
