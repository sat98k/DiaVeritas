# =============================================================================
# src/retrieval/bm25_index.py
#
# BM25 lexical retrieval index using rank-bm25 (BM25Okapi).
#
# Why BM25 alongside dense retrieval?
#   Dense retrieval is strong on paraphrase and semantic similarity.
#   BM25 excels at exact terminology: drug names, biomarkers, abbreviations.
#   Example: a query mentioning "SGLT2" will get exact matches via BM25
#   even if the dense embedding doesn't align perfectly.
#
# The BM25 index is built from all chunk texts, tokenized by whitespace.
# It is serialized to disk (pickle) after building so it doesn't need
# rebuilding on every query.
#
# IMPORTANT: BM25 operates over the chunk corpus, not the raw papers.
# =============================================================================

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

from src.preprocessing.chunker import Chunk


# ---------------------------------------------------------------------------
# Simple biomedical tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25.

    Uses whitespace + punctuation splitting with lowercasing.
    Keeps hyphenated terms together (e.g., "GLP-1", "HbA1c") because
    these are meaningful in biomedical text.
    """
    # Lowercase
    text = text.lower()
    # Split on whitespace and common separators, keep hyphens within words
    tokens = re.findall(r"\b[\w][\w\-\.]*[\w]\b|\b\w\b", text)
    # Filter very short tokens
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    BM25 lexical retrieval index over DiaVeritas evidence chunks.

    Build:
        index = BM25Index()
        index.build(chunks)
        index.save(path)

    Query:
        index = BM25Index.load(path)
        results = index.query("metformin cardiovascular outcomes", n_results=50)
    """

    def __init__(self):
        self._bm25 = None
        self._chunks: List[Chunk] = []
        self._tokenized_corpus: List[List[str]] = []

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build the BM25 index from a list of chunks.

        Args:
            chunks: All chunks in the corpus (from load_chunks).
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 is required. Run: py -m pip install rank-bm25")

        logger.info(f"Building BM25 index over {len(chunks)} chunks...")
        self._chunks = chunks
        self._tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built.")

    def query(
        self,
        query_text: str,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Run a BM25 query and return the top-n results.

        Args:
            query_text: The raw query string.
            n_results: Number of results to return.

        Returns:
            List of dicts:
            {
                "chunk_id": str,
                "text": str,
                "score": float,   # BM25 score (higher is better)
                "chunk": Chunk,
                "rank": int
            }
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top n_results by score
        n_results = min(n_results, len(self._chunks))
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:n_results]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue  # BM25 score of 0 means no match
            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": score,
                "chunk": chunk,
                "rank": rank + 1,
            })

        return results

    def save(self, path: Path) -> None:
        """Serialize the BM25 index to disk using pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load a serialized BM25 index from disk."""
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. "
                "Run build_index.py --step index first."
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)
        index = cls()
        index._bm25 = payload["bm25"]
        index._chunks = payload["chunks"]
        index._tokenized_corpus = payload["tokenized_corpus"]
        logger.info(f"BM25 index loaded: {len(index._chunks)} chunks from {path}")
        return index

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)
