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

import re
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
        import torch
        _nli_pipeline = hf_pipeline(
            "text-classification",
            model=target,
            dtype=torch.float32,
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

_OUTCOME_CUES = {
    "result", "conclusion", "effect", "reduc", "increas", "signific",
    "efficac", "superior", "favor", "improv", "hazard ratio", "lowered",
    "benefit", "risk", "mortality", "associated", "difference", "demonstrated"
}

_COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


_DESIGN_PENALTIES = {
    "primary outcome was", "primary end point was", "primary endpoint was",
    "randomly assigned", "study design", "were enrolled", "inclusion criteria",
    "exclusion criteria", "to investigate whether", "to determine whether"
}


def select_salient_premise_sentences(
    text: str,
    hypothesis: str,
    max_candidates: int = 3,
) -> List[str]:
    """
    Select the most salient candidate sentences from an evidence passage against a hypothesis.

    Cross-encoder NLI models (MNLI-trained) operate on sentence-pair attention.
    Passing a 400-word paragraph causes background demographic and protocol text to dilute
    attention and wash out decisive clinical findings into NEUTRAL.
    Scoring individual sentences by lexical overlap and clinical outcome indicators ensures
    the NLI model evaluates the exact assertions made by the study.
    """
    if not text or len(text.strip()) < 20:
        return [text.strip()] if text and text.strip() else []

    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 20
    ]
    if not raw_sentences:
        return [text[:500].strip()]
    if len(raw_sentences) <= max_candidates:
        return raw_sentences

    hyp_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", hypothesis.lower())) - _COMMON_STOPWORDS

    scored: List[Tuple[float, str]] = []
    total_sents = len(raw_sentences)
    for idx, s in enumerate(raw_sentences):
        s_lower = s.lower()
        s_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", s_lower))
        overlap = len(hyp_tokens & s_tokens)
        cue_bonus = 2.5 if any(cue in s_lower for cue in _OUTCOME_CUES) else 0.0
        design_penalty = 4.0 if any(dp in s_lower for dp in _DESIGN_PENALTIES) else 0.0
        pos_bonus = 1.0 if idx >= total_sents // 2 else 0.0
        score = overlap * 2.0 + cue_bonus + pos_bonus - design_penalty
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_candidates]]


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

    candidates = select_salient_premise_sentences(premise, hypothesis, max_candidates=2)
    if not candidates:
        candidates = [premise[:500]]

    pairs = [{"text": c[:500], "text_pair": hypothesis[:300]} for c in candidates]
    try:
        outputs = pipe(pairs, batch_size=len(pairs))
    except Exception:
        outputs = [pipe(p) for p in pairs]

    # If pipeline returned a single item's score list: [{label: ..., score: ...}, ...]
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict) and "label" in outputs[0]:
        outputs = [outputs]

    parsed_candidates = []
    for cand_text, output in zip(candidates, outputs):
        out_list = output if isinstance(output, list) else [output]
        scores_by_canonical = {}
        for item in out_list:
            canonical = _LABEL_MAP.get(item["label"].lower(), "NEUTRAL")
            scores_by_canonical[canonical] = float(item["score"])
        for lbl in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
            scores_by_canonical.setdefault(lbl, 0.0)
        top_canonical = max(scores_by_canonical, key=scores_by_canonical.get)
        parsed_candidates.append({
            "text": cand_text,
            "label": top_canonical,
            "confidence": scores_by_canonical[top_canonical],
            "scores": scores_by_canonical,
        })

    # Pick candidate with strongest non-neutral signal if >= 0.50
    decisive = [c for c in parsed_candidates if c["label"] in ("ENTAILMENT", "CONTRADICTION") and c["confidence"] >= 0.50]
    best = max(decisive, key=lambda x: x["confidence"]) if decisive else max(parsed_candidates, key=lambda x: x["confidence"])

    return NLIResult(
        premise_chunk_id=premise_chunk_id,
        hypothesis_text=hypothesis,
        label=best["label"],
        confidence=round(best["confidence"], 4),
        all_scores={k: round(v, 4) for k, v in best["scores"].items()},
        premise_text=best["text"][:300],
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

    Extracts salient premise candidate sentences per chunk and batches inference
    across all candidates in a single high-throughput forward pass.

    Args:
        query_claim_text: A string representing the core query claim.
        evidence_claims: List of StructuredClaim objects from claim_extractor.
        model_name: Override NLI model.

    Returns:
        List of NLIResult, one per evidence claim.
    """
    logger.info(
        f"Running NLI: {len(evidence_claims)} evidence claims vs query hypothesis."
    )
    if not evidence_claims:
        return []

    pipe = _load_nli_model(model_name)

    # Prepare candidate sentences per chunk (top 3 salient sentences)
    claim_candidate_map: List[Tuple[str, List[str]]] = []
    flat_inputs = []

    for claim in evidence_claims:
        candidates = select_salient_premise_sentences(
            claim.raw_text, query_claim_text, max_candidates=3
        )
        if not candidates:
            fallback = claim.to_claim_string() if len(claim.to_claim_string()) >= 10 else claim.raw_text[:500]
            candidates = [fallback]
        claim_candidate_map.append((claim.chunk_id, candidates))
        for cand in candidates:
            flat_inputs.append({"text": cand[:500], "text_pair": query_claim_text[:300]})

    batch_outputs = pipe(flat_inputs, batch_size=len(flat_inputs))
    if isinstance(batch_outputs, list) and len(batch_outputs) > 0 and isinstance(batch_outputs[0], dict) and "label" in batch_outputs[0]:
        batch_outputs = [batch_outputs]

    # Reconstruct results per chunk
    out_idx = 0
    results: List[NLIResult] = []

    for chunk_id, candidates in claim_candidate_map:
        chunk_cand_results = []
        for cand in candidates:
            output = batch_outputs[out_idx]
            out_idx += 1
            out_list = output if isinstance(output, list) else [output]
            scores_by_canonical = {}
            for item in out_list:
                canonical = _LABEL_MAP.get(item["label"].lower(), "NEUTRAL")
                scores_by_canonical[canonical] = float(item["score"])
            for lbl in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
                scores_by_canonical.setdefault(lbl, 0.0)
            top_canonical = max(scores_by_canonical, key=scores_by_canonical.get)
            chunk_cand_results.append({
                "text": cand,
                "label": top_canonical,
                "confidence": scores_by_canonical[top_canonical],
                "scores": scores_by_canonical,
            })

        # Prefer non-neutral candidate when there is meaningful directional evidence
        decisive = []
        for c in chunk_cand_results:
            ent = c["scores"].get("ENTAILMENT", 0.0)
            con = c["scores"].get("CONTRADICTION", 0.0)
            if c["label"] in ("ENTAILMENT", "CONTRADICTION") and c["confidence"] >= 0.40:
                decisive.append(c)
            elif ent >= 0.35 and ent > 2.0 * con:
                c_adjusted = dict(c)
                c_adjusted["label"] = "ENTAILMENT"
                c_adjusted["confidence"] = ent
                decisive.append(c_adjusted)
            elif con >= 0.35 and con > 2.0 * ent:
                c_adjusted = dict(c)
                c_adjusted["label"] = "CONTRADICTION"
                c_adjusted["confidence"] = con
                decisive.append(c_adjusted)

        best = max(decisive, key=lambda x: x["confidence"]) if decisive else max(chunk_cand_results, key=lambda x: x["confidence"])

        res = NLIResult(
            premise_chunk_id=chunk_id,
            hypothesis_text=query_claim_text,
            label=best["label"],
            confidence=round(best["confidence"], 4),
            all_scores={k: round(v, 4) for k, v in best["scores"].items()},
            premise_text=best["text"][:300],
        )
        results.append(res)
        logger.debug(f"NLI [{chunk_id}]: {res.label} ({res.confidence:.3f}) - '{res.premise_text[:80]}...'")

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
