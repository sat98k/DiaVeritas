# =============================================================================
# DiaVeritas — Central Configuration
#
# Edit this file to change experimental parameters.
# All important thresholds and model choices live here.
# =============================================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # LLM
    # -------------------------------------------------------------------------
    llm_provider: str = Field(default="groq", description="openai | groq | google | anthropic | deepseek | local")
    llm_model: str = Field(default="qwen/qwen3.8-27b")
    openai_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    deepseek_api_key: str = Field(default="")
    huggingfacehub_api_token: str = Field(default="")

    # -------------------------------------------------------------------------
    # NCBI / PubMed corpus collection
    # -------------------------------------------------------------------------
    ncbi_api_key: str = Field(default="")
    ncbi_email: str = Field(default="your_email@example.com")

    # -------------------------------------------------------------------------
    # Embedding model
    # -------------------------------------------------------------------------
    embedding_model: str = Field(
        default="pritamdeka/S-PubMedBert-MS-MARCO",
        description=(
            "Primary: pritamdeka/S-PubMedBert-MS-MARCO (~420MB, biomedical SBERT). "
            "Fallback: all-MiniLM-L6-v2 (~80MB, general-purpose)."
        ),
    )
    embedding_batch_size: int = Field(default=32)

    # -------------------------------------------------------------------------
    # Cross-encoder reranker
    # -------------------------------------------------------------------------
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Lightweight cross-encoder; replace with biomedical-specific if available.",
    )

    # -------------------------------------------------------------------------
    # Biomedical NLI model
    # -------------------------------------------------------------------------
    nli_model: str = Field(
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        description=(
            "General-purpose NLI as CPU-feasible default. "
            "Preferred biomedical: 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext' "
            "fine-tuned on NLI (if available/feasible). "
            "SciFact-trained: 'tomaarsen/bert-base-cased-nli-scifact'."
        ),
    )

    # -------------------------------------------------------------------------
    # Vector store
    # -------------------------------------------------------------------------
    vector_store_backend: str = Field(default="chromadb", description="chromadb | faiss")
    vector_store_path: Path = Field(default=ROOT_DIR / "vector_store")
    chroma_collection_name: str = Field(default="diaveritias_chunks")

    # -------------------------------------------------------------------------
    # Chunking parameters (experimental — easy to tune)
    # -------------------------------------------------------------------------
    chunk_size_tokens: int = Field(
        default=400,
        description="Target chunk size in tokens. Range: 300–500. Experimental.",
    )
    chunk_overlap_percent: int = Field(
        default=15,
        description="Overlap between consecutive chunks as % of chunk_size. Range: 10–20%.",
    )

    # -------------------------------------------------------------------------
    # Retrieval parameters
    # -------------------------------------------------------------------------
    dense_retrieval_candidates: int = Field(
        default=50,
        description="Number of dense retrieval candidates before RRF.",
    )
    bm25_candidates: int = Field(
        default=50,
        description="Number of BM25 candidates before RRF.",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant k. Standard default is 60.",
    )
    reranker_top_k: int = Field(
        default=10,
        description="Final top-k after cross-encoder reranking.",
    )

    # -------------------------------------------------------------------------
    # NLI thresholds
    # -------------------------------------------------------------------------
    nli_contradiction_threshold: float = Field(
        default=0.5,
        description="Minimum NLI score to flag a contradiction relationship.",
    )
    nli_entailment_threshold: float = Field(
        default=0.5,
        description="Minimum NLI score to flag an entailment relationship.",
    )

    # -------------------------------------------------------------------------
    # Evidence status
    # -------------------------------------------------------------------------
    supported_min_entailment_ratio: float = Field(
        default=0.6,
        description="Fraction of NLI pairs that must be entailment to call SUPPORTED.",
    )
    refuted_min_contradiction_ratio: float = Field(
        default=0.6,
        description="Fraction of NLI pairs that must be contradiction to call REFUTED.",
    )

    # -------------------------------------------------------------------------
    # Data paths
    # -------------------------------------------------------------------------
    data_dir: Path = Field(default=ROOT_DIR / "data")
    raw_dir: Path = Field(default=ROOT_DIR / "data" / "raw")
    processed_dir: Path = Field(default=ROOT_DIR / "data" / "processed")
    metadata_dir: Path = Field(default=ROOT_DIR / "data" / "metadata")

    # -------------------------------------------------------------------------
    # Baseline mode
    # -------------------------------------------------------------------------
    baseline_mode: bool = Field(
        default=False,
        description="If True, skip NLI/context/evidence pipeline and use plain RAG → LLM.",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO")


# Module-level singleton
settings = Settings()
