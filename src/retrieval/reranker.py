# =============================================================================
# src/retrieval/reranker.py
#
# Cross-encoder reranker for the candidate evidence pool.
#
# Pipeline position:
#   Dense+BM25 → RRF (50 candidates) → Cross-encoder → Top-k (default 10)
#
# Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - ~85MB, fast on CPU
#   - Strong passage reranking for question-answer relevance
#   - Not biomedical-specific, but the most practical CPU-feasible option
#
# Cross-encoder scores query-passage pairs jointly (not independently),
# which captures relevance more accurately than bi-encoder cosine similarity.
#
# The reranker scores are stored on each result dict for the debug view.
# =============================================================================

from __future__ import annotations

from typing import List, Dict, Any, Optional

from loguru import logger

from src.config import settings


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for passage reranking.

    Usage:
        reranker = CrossEncoderReranker()
        top_k = reranker.rerank(query, candidates, top_k=10)
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.reranker_model
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder reranker: {self._model_name}")
            self._model = CrossEncoder(self._model_name, max_length=512)
            logger.info("Cross-encoder loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cross-encoder '{self._model_name}': {e}"
            ) from e

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate results using the cross-encoder.

        Args:
            query: The user query.
            candidates: List of result dicts from HybridRetriever.retrieve().
                        Each must have a "text" key.
            top_k: Number of results to return after reranking.
                   Defaults to settings.reranker_top_k.

        Returns:
            Top-k results sorted by cross-encoder score descending.
            Each result gets a "reranker_score" field added.
        """
        self._load()

        if not candidates:
            return []

        top_k = top_k or settings.reranker_top_k

        # Limit cross-encoder candidates to top-25 from RRF for sub-second CPU latency
        candidates_to_rerank = candidates[:25]

        # Build query-passage pairs for cross-encoder
        pairs = [[query, item["text"]] for item in candidates_to_rerank]

        logger.info(f"Reranking {len(candidates_to_rerank)} candidates with cross-encoder...")
        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Attach scores and sort
        reranked = []
        for item, score in zip(candidates_to_rerank, scores):
            item_copy = item.copy()
            item_copy["reranker_score"] = float(score)
            reranked.append(item_copy)

        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Assign final reranker ranks
        for rank, item in enumerate(reranked):
            item["reranker_rank"] = rank + 1

        result = reranked[:top_k]
        logger.info(
            f"Reranking complete. Returned top-{len(result)} "
            f"(scores: {result[0]['reranker_score']:.3f} – "
            f"{result[-1]['reranker_score']:.3f})"
            if result else "Reranking returned 0 results."
        )
        return result

    @property
    def model_name(self) -> str:
        return self._model_name
