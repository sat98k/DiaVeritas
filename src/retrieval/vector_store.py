# =============================================================================
# src/retrieval/vector_store.py
#
# ChromaDB wrapper for dense vector storage and retrieval.
#
# Stores chunk embeddings + full metadata so that retrieved chunks carry
# all their source information (paper_id, section, title, year, etc.).
#
# ChromaDB runs in-process (no separate server needed) with local persistence
# at settings.vector_store_path. This is appropriate for a research prototype
# with a corpus of tens to hundreds of papers.
#
# FAISS alternative: uncomment in requirements.txt and swap this module.
# =============================================================================

from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

import numpy as np
from loguru import logger

from src.config import settings
from src.preprocessing.chunker import Chunk


class VectorStore:
    """
    ChromaDB-backed vector store for DiaVeritas evidence chunks.

    The collection stores:
    - embeddings (dense vectors)
    - documents (chunk text, for retrieval inspection)
    - metadata (all chunk fields except text, stored as flat key-value pairs)
    """

    def __init__(self, fresh: bool = False):
        """
        Args:
            fresh: If True, delete and recreate the collection.
        """
        self._collection = None
        self._fresh = fresh
        self._load()

    def _load(self):
        """Initialize ChromaDB client and get or create the collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb is required. Run: py -m pip install chromadb")

        persist_path = str(settings.vector_store_path)
        logger.info(f"ChromaDB: persist_path={persist_path}")

        self._client = chromadb.PersistentClient(path=persist_path)

        if self._fresh:
            try:
                self._client.delete_collection(settings.chroma_collection_name)
                logger.info(f"Deleted existing collection '{settings.chroma_collection_name}'")
            except Exception:
                pass  # Collection didn't exist

        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{settings.chroma_collection_name}' "
            f"loaded: {self._collection.count()} items."
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their pre-computed embeddings to the collection.

        ChromaDB metadata values must be str, int, float, or bool.
        We flatten lists (authors, entities) to JSON strings.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        # Ensure IDs are unique to prevent DuplicateIDError
        seen_ids = set()
        dedup_chunks = []
        dedup_indices = []
        for idx, c in enumerate(chunks):
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                dedup_chunks.append(c)
                dedup_indices.append(idx)

        if len(dedup_chunks) < len(chunks):
            logger.info(
                f"Deduplicated {len(chunks) - len(dedup_chunks)} duplicate chunk IDs before adding to ChromaDB."
            )
            chunks = dedup_chunks
            embeddings = embeddings[dedup_indices]

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings_list = embeddings.tolist()
        metadatas = [_chunk_to_metadata(c) for c in chunks]

        # ChromaDB has a max batch size; we add in batches
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            self._collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                embeddings=embeddings_list[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
            logger.debug(f"Added batch {i}–{i + batch_size}")

        logger.info(f"Added {len(chunks)} chunks to vector store.")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 50,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for nearest neighbors.

        Args:
            query_embedding: 1D numpy array (the encoded query).
            n_results: Number of results to return.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of dicts, each containing:
            {
                "chunk_id": str,
                "text": str,
                "score": float,  # cosine similarity (higher is better)
                "metadata": dict,
                "chunk": Chunk   # reconstituted Chunk object
            }
        """
        actual_count = self._collection.count()
        if actual_count == 0:
            logger.warning("Vector store is empty. Run build_index.py first.")
            return []

        n_results = min(n_results, actual_count)

        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        items = []
        for i, (chunk_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # ChromaDB cosine distance: score = 1 - distance
            score = 1.0 - dist
            chunk = _metadata_to_chunk(chunk_id, doc, meta)
            items.append({
                "chunk_id": chunk_id,
                "text": doc,
                "score": score,
                "metadata": meta,
                "chunk": chunk,
                "rank": i + 1,
            })

        return items

    def get_all_ids(self) -> List[str]:
        """Return all chunk IDs currently in the collection."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=[])
        return result["ids"]

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Metadata serialization helpers
# ---------------------------------------------------------------------------

def _chunk_to_metadata(chunk: Chunk) -> Dict[str, Any]:
    """
    Convert a Chunk to a flat dict suitable for ChromaDB metadata.
    ChromaDB requires all values to be str, int, float, or bool.
    """
    return {
        "paper_id": chunk.paper_id,
        "pmc_id": chunk.pmc_id,
        "title": chunk.title,
        "authors": json.dumps(chunk.authors),  # List → JSON string
        "year": chunk.year,
        "journal": chunk.journal,
        "doi": chunk.doi,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "study_type": chunk.study_type,
        "source": chunk.source,
        "token_count": chunk.token_count,
        "entities": json.dumps(chunk.entities),  # Dict → JSON string
    }


def _metadata_to_chunk(chunk_id: str, text: str, meta: Dict) -> Chunk:
    """Reconstruct a Chunk from ChromaDB metadata + document text."""
    return Chunk(
        chunk_id=chunk_id,
        paper_id=meta.get("paper_id", "Unknown"),
        pmc_id=meta.get("pmc_id", "Unknown"),
        title=meta.get("title", "Not reported"),
        authors=json.loads(meta.get("authors", '["Not reported"]')),
        year=meta.get("year", "Unknown"),
        journal=meta.get("journal", "Not reported"),
        doi=meta.get("doi", "Not reported"),
        section=meta.get("section", "Unknown"),
        chunk_index=int(meta.get("chunk_index", 0)),
        study_type=meta.get("study_type", "Not reported"),
        source=meta.get("source", "unknown"),
        text=text,
        token_count=int(meta.get("token_count", 0)),
        entities=json.loads(meta.get("entities", "{}")),
    )
