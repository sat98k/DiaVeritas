# =============================================================================
# src/nli/nli_classifier.py
#
# Biomedical Natural Language Inference (NLI) classifier.
#
# Classifies the relationship between a query claim and an evidence claim as:
#   - ENTAILMENT:    The evidence supports the query claim
#   - CONTRADICTION: The evidence contradicts the query claim
#   - NEUTRAL:       The evidence neither supports nor contradicts
#
# Default model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
#   - CPU-feasible (~700MB)
#   - Strong general-purpose NLI
#   - Handles scientific language well
#
# Alternative: tomaarsen/bert-base-cased-nli-scifact
#   - SciFact-trained (biomedical claim verification dataset)
#   - Smaller, ~200MB
#   - Set NLI_MODEL in .env to switch
#
# IMPORTANT:
#   NLI detects semantic relationships, NOT medical truth.
#   A CONTRADICTION label means the claims make opposite assertions.
#   It does NOT mean one paper is wrong.
#   Context analysis (contradiction_analyzer.py) interprets WHY they differ.
#
#   NEUTRAL does NOT mean false or uninformative.
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from loguru import logger

from src.config import settings
from src.claims.claim_extractor import StructuredClaim


# ---------------------------------------------------------------------------
# NLI result structure
# ---------------------------------------------------------------------------

@dataclass
class NLIResult:
    """
    Result of a single NLI inference.

    premise: The evidence claim (what the paper says)
    hypothesis: The query claim (what we're asking about)
    label: ENTAILMENT | CONTRADICTION | NEUTRAL
    confidence: float in [0, 1] — model softmax score for the predicted label
    all_scores: dict with scores for all three labels
    """
    premise_chunk_id: str
    hypothesis_text: str
    label: str                        # ENTAILMENT | CONTRADICTION | NEUTRAL
    confidence: float
    all_scores: Dict[str, float]      # {"ENTAILMENT": 0.7, "CONTRADICTION": 0.1, "NEUTRAL": 0.2}
    premise_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_entailment(self) -> bool:
        return self.label == "ENTAILMENT" and self.confidence >= settings.nli_entailment_threshold

    @property
    def is_contradiction(self) -> bool:
        return self.label == "CONTRADICTION" and self.confidence >= settings.nli_contradiction_threshold

    @property
    def is_neutral(self) -> bool:
        return self.label == "NEUTRAL"


# ---------------------------------------------------------------------------
# NLI model (lazy-loaded)
# ---------------------------------------------------------------------------

_nli_pipeline = None
_nli_model_name_loaded = None

# Label mapping: different models use different label strings
# We normalize to ENTAILMENT / CONTRADICTION / NEUTRAL
_LABEL_MAP = {
    "entailment": "ENTAILMENT",
    "contradiction": "CONTRADICTION",
    "neutral": "NEUTRAL",
    "not_entailment": "CONTRADICTION",
    "supports": "ENTAILMENT",
    "refutes": "CONTRADICTION",
}


def _load_nli_model(model_name: Optional[str] = None):
    """Lazy-load the NLI pipeline."""
    global _nli_pipeline, _nli_model_name_loaded

    target = model_name or settings.nli_model
    if _nli_pipeline is not None and _nli_model_name_loaded == target:
        return _nli_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading NLI model: {target}")
        logger.info("(First run may download ~700MB — this may take a few minutes)")
        _nli_pipeline = hf_pipeline(
            "text-classification",
            model=target,
            top_k=None,  # return all label scores
            device=-1,   # CPU
            truncation=True,
            max_length=512,
        )
        _nli_model_name_loaded = target
        logger.info(f"NLI model loaded: {target}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load NLI model '{target}': {e}\n"
            f"Try: py -m pip install transformers"
        ) from e

    return _nli_pipeline


# ---------------------------------------------------------------------------
# Core NLI function
# ---------------------------------------------------------------------------

def classify_nli(
    premise: str,
    hypothesis: str,
    premise_chunk_id: str = "unknown",
    model_name: Optional[str] = None,
) -> NLIResult:
    """
    Run NLI on a premise-hypothesis pair.

    Args:
        premise: The evidence passage or extracted claim (what the paper says).
        hypothesis: The query claim (what we're verifying).
        premise_chunk_id: Chunk ID for traceability.
        model_name: Override NLI model. Defaults to settings.nli_model.

    Returns:
        NLIResult with label and confidence scores.
    """
    pipe = _load_nli_model(model_name)

    # Format as required by text-classification NLI models
    # Input format: "premise [SEP] hypothesis" — handled by the model's tokenizer
    # Some models expect the pair as a single string; HF pipeline handles this.
    try:
        outputs = pipe(
            {"text": premise[:900], "text_pair": hypothesis[:300]},
        )
    except TypeError:
        # Some pipeline versions need a list
        outputs = pipe(
            [{"text": premise[:900], "text_pair": hypothesis[:300]}]
        )[0]

    # Parse scores
    all_scores: Dict[str, float] = {}
    for item in outputs:
        label_raw = item["label"].lower()
        normalized = _LABEL_MAP.get(label_raw, label_raw.upper())
        all_scores[normalized] = float(item["score"])

    # Ensure all three labels exist
    for lbl in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
        all_scores.setdefault(lbl, 0.0)

    # Winner
    predicted_label = max(all_scores, key=all_scores.get)
    confidence = all_scores[predicted_label]

    return NLIResult(
        premise_chunk_id=premise_chunk_id,
        hypothesis_text=hypothesis,
        label=predicted_label,
        confidence=confidence,
        all_scores=all_scores,
        premise_text=premise[:200],
    )


# ---------------------------------------------------------------------------
# Batch NLI over multiple evidence chunks
# ---------------------------------------------------------------------------

def classify_evidence_against_query(
    query_claim_text: str,
    evidence_claims: List[StructuredClaim],
    model_name: Optional[str] = None,
) -> List[NLIResult]:
    """
    Run NLI for each evidence claim against the query claim.

    The query claim is used as the HYPOTHESIS.
    Each evidence claim is the PREMISE.

    This framing asks: "Does this evidence passage support or contradict
    the query claim?"

    Args:
        query_claim_text: A string representing the core query claim
                          (e.g., from query_normalizer's normalized_text
                          or a structured summary of what the user is asking).
        evidence_claims: List of StructuredClaim objects from claim_extractor.
        model_name: Override NLI model.

    Returns:
        List of NLIResult, one per evidence claim.
    """
    logger.info(
        f"Running NLI: {len(evidence_claims)} evidence claims vs query hypothesis."
    )

    results: List[NLIResult] = []
    for claim in evidence_claims:
        # Use the structured claim string as premise
        premise_text = claim.to_claim_string()
        if len(premise_text) < 10:
            # Fallback to raw text if claim string is too short
            premise_text = claim.raw_text[:600]

        result = classify_nli(
            premise=premise_text,
            hypothesis=query_claim_text,
            premise_chunk_id=claim.chunk_id,
            model_name=model_name,
        )
        results.append(result)
        logger.debug(
            f"NLI [{claim.chunk_id}]: {result.label} ({result.confidence:.3f})"
        )

    return results


# ---------------------------------------------------------------------------
# NLI summary statistics
# ---------------------------------------------------------------------------

def summarize_nli_results(results: List[NLIResult]) -> Dict[str, Any]:
    """
    Summarize NLI results across all evidence claims.

    Returns counts, ratios, and lists of entailment/contradiction/neutral results.
    """
    counts = {"ENTAILMENT": 0, "CONTRADICTION": 0, "NEUTRAL": 0}
    for r in results:
        counts[r.label] = counts.get(r.label, 0) + 1

    total = len(results)
    ratios = {k: round(v / total, 3) if total > 0 else 0.0 for k, v in counts.items()}

    entailments = [r for r in results if r.label == "ENTAILMENT"]
    contradictions = [r for r in results if r.label == "CONTRADICTION"]
    neutrals = [r for r in results if r.label == "NEUTRAL"]

    avg_confidence = (
        sum(r.confidence for r in results) / total if total > 0 else 0.0
    )

    return {
        "total": total,
        "counts": counts,
        "ratios": ratios,
        "avg_confidence": round(avg_confidence, 3),
        "entailments": entailments,
        "contradictions": contradictions,
        "neutrals": neutrals,
    }
