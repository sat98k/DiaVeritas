# =============================================================================
# src/retrieval/embedder.py
#
# Sentence-transformer encoder for dense retrieval.
#
# Default model: pritamdeka/S-PubMedBert-MS-MARCO
#   - Biomedical SBERT trained on PubMed + MS-MARCO
#   - ~420MB download on first use
#   - CPU-feasible: batch encoding for index building, fast for query encoding
#
# Fallback: all-MiniLM-L6-v2 (~80MB, general-purpose)
#   - Set EMBEDDING_MODEL=all-MiniLM-L6-v2 in .env for lighter footprint
#
# Interface is intentionally simple: encode(texts) → numpy array of vectors.
# =============================================================================

from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from src.config import settings


class Embedder:
    """
    Wraps a sentence-transformers model for text encoding.

    Lazy-loads the model on first use to keep startup time fast.
    """

    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None):
        """
        Args:
            model_name: HuggingFace model name. Defaults to settings.embedding_model.
            batch_size: Encoding batch size. Defaults to settings.embedding_batch_size.
        """
        self._model_name = model_name or settings.embedding_model
        self._batch_size = batch_size or settings.embedding_batch_size
        self._model = None

    def _load(self):
        """Lazy-load the sentence transformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_name}': {e}\n"
                f"Try: py -m pip install sentence-transformers"
            ) from e

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Args:
            texts: List of strings to encode.
            normalize: If True, L2-normalize vectors (recommended for cosine similarity).
            show_progress: Show tqdm progress bar during batch encoding.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._load()
        if not texts:
            return np.array([])

        logger.info(f"Encoding {len(texts)} texts (batch_size={self._batch_size})...")
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.info(f"Encoding complete. Shape: {embeddings.shape}")
        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query string. Optimized path for online inference.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        self._load()
        vec = self._model.encode(
            query,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name
