# =============================================================================
# src/claims/claim_extractor.py
#
# Extracts structured claims from evidence passages.
#
# A "claim" is a structured representation of the key assertion in an
# evidence chunk, capturing:
#   - Intervention
#   - Population
#   - Comparator
#   - Outcome
#   - Direction of effect (Reduction / Increase / No significant change / Unclear)
#   - Magnitude (if reported)
#   - Duration (if reported)
#   - Study context
#
# Approach:
#   Uses the configured LLM (via llm_client) with a structured prompt
#   to extract claims. This is more accurate than rule-based IE for the
#   diversity of biomedical expression styles.
#
#   Falls back to a keyword-heuristic extractor if no LLM is available.
#
# IMPORTANT:
#   The LLM is instructed NOT to invent information.
#   Missing fields are explicitly returned as "Not reported".
#   The claim extractor does NOT make clinical judgments.
# =============================================================================

from __future__ import annotations

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from loguru import logger

from src.preprocessing.chunker import Chunk


# ---------------------------------------------------------------------------
# Claim data structure
# ---------------------------------------------------------------------------

@dataclass
class StructuredClaim:
    """
    Structured representation of a claim extracted from an evidence passage.

    All fields default to "Not reported" if unavailable.
    The LLM / extractor must NOT invent clinical information.
    """
    chunk_id: str
    paper_id: str
    section: str
    intervention: str = "Not reported"
    population: str = "Not reported"
    comparator: str = "Not reported"
    outcome: str = "Not reported"
    direction: str = "Unclear"       # Reduction | Increase | No significant change | Unclear
    magnitude: str = "Not reported"  # e.g., "HbA1c reduced by 1.2%"
    duration: str = "Not reported"   # e.g., "24 weeks"
    study_context: str = "Not reported"  # e.g., "RCT, n=500, high-risk T2D"
    raw_text: str = ""               # The original evidence passage

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_claim_string(self) -> str:
        """Human-readable one-line claim summary for NLI input."""
        parts = []
        if self.intervention != "Not reported":
            parts.append(self.intervention)
        if self.direction not in ("Not reported", "Unclear"):
            parts.append(self.direction.lower())
        if self.outcome != "Not reported":
            parts.append(self.outcome)
        if self.population != "Not reported":
            parts.append(f"in {self.population}")
        if self.comparator != "Not reported":
            parts.append(f"vs {self.comparator}")
        return " ".join(parts) if parts else self.raw_text[:200]


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_PROMPT = """You are a biomedical information extraction assistant analyzing a clinical research passage.

Extract the primary clinical claim from the following evidence passage. Return ONLY a JSON object with these exact keys:
- "intervention": the drug/treatment being studied (or "Not reported")
- "population": the patient population (or "Not reported")
- "comparator": what the intervention is compared to, e.g., placebo, standard care (or "Not reported")
- "outcome": the primary outcome being measured (or "Not reported")
- "direction": one of: "Reduction", "Increase", "No significant change", "Unclear"
- "magnitude": the reported effect size or change (or "Not reported")
- "duration": the study duration or follow-up period (or "Not reported")
- "study_context": brief study design info like RCT, sample size, population risk level (or "Not reported")

CRITICAL RULES:
1. Do NOT invent information not present in the passage.
2. Use "Not reported" for any field not mentioned in the passage.
3. Do NOT add clinical judgments or interpretations beyond what is stated.
4. Return ONLY the JSON object, no other text.

Evidence passage:
{text}

JSON:"""


def extract_claim_with_llm(chunk: Chunk, llm_client) -> StructuredClaim:
    """
    Extract a structured claim using the configured LLM.

    Args:
        chunk: The evidence chunk to extract from.
        llm_client: An initialized LLMClient instance.

    Returns:
        StructuredClaim with extracted fields.
    """
    prompt = _CLAIM_EXTRACTION_PROMPT.format(text=chunk.text[:1500])

    try:
        response = llm_client.complete(prompt, max_tokens=400, temperature=0.0)
        # Parse JSON from response
        # Sometimes LLMs add markdown code blocks — strip them
        json_text = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
        data = json.loads(json_text)

        return StructuredClaim(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            section=chunk.section,
            intervention=_safe_field(data, "intervention"),
            population=_safe_field(data, "population"),
            comparator=_safe_field(data, "comparator"),
            outcome=_safe_field(data, "outcome"),
            direction=_safe_direction(data.get("direction", "Unclear")),
            magnitude=_safe_field(data, "magnitude"),
            duration=_safe_field(data, "duration"),
            study_context=_safe_field(data, "study_context"),
            raw_text=chunk.text,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error in claim extraction for {chunk.chunk_id}: {e}")
        return _fallback_claim(chunk)
    except Exception as e:
        logger.warning(f"LLM claim extraction failed for {chunk.chunk_id}: {e}")
        return _fallback_claim(chunk)


# ---------------------------------------------------------------------------
# Keyword-heuristic fallback
# ---------------------------------------------------------------------------

_DIRECTION_POSITIVE = {"reduc", "decreas", "lower", "improve", "benefit", "signif"}
_DIRECTION_NEGATIVE = {"increas", "worsen", "harm", "risk", "elevat"}
_DIRECTION_NULL = {"no significant", "not significant", "did not", "no effect", "no difference"}


def extract_claim_heuristic(chunk: Chunk) -> StructuredClaim:
    """
    Keyword-heuristic claim extractor (fallback when LLM unavailable).

    Less accurate than LLM extraction but fully deterministic and fast.
    Uses chunk entities and simple text patterns.
    """
    text_lower = chunk.text.lower()

    # Intervention: first drug entity
    drugs = chunk.entities.get("drugs", [])
    intervention = drugs[0] if drugs else "Not reported"

    # Outcome: first outcome entity
    outcomes = chunk.entities.get("outcomes", [])
    outcome = outcomes[0] if outcomes else "Not reported"

    # Biomarker as fallback outcome
    if outcome == "Not reported":
        biomarkers = chunk.entities.get("biomarkers", [])
        outcome = biomarkers[0] if biomarkers else "Not reported"

    # Direction heuristic
    direction = "Unclear"
    if any(kw in text_lower for kw in _DIRECTION_NULL):
        direction = "No significant change"
    elif any(kw in text_lower for kw in _DIRECTION_POSITIVE):
        direction = "Reduction"
    elif any(kw in text_lower for kw in _DIRECTION_NEGATIVE):
        direction = "Increase"

    # Population from diseases
    diseases = chunk.entities.get("diseases", [])
    population = diseases[0] if diseases else "Not reported"

    return StructuredClaim(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        section=chunk.section,
        intervention=intervention,
        population=population,
        comparator="Not reported",
        outcome=outcome,
        direction=direction,
        magnitude="Not reported",
        duration="Not reported",
        study_context=f"{chunk.study_type}" if chunk.study_type != "Not reported" else "Not reported",
        raw_text=chunk.text,
    )


# ---------------------------------------------------------------------------
# Main extraction dispatcher
# ---------------------------------------------------------------------------

def extract_claims(
    chunks: List[Chunk],
    llm_client=None,
) -> List[StructuredClaim]:
    """
    Extract structured claims from a list of chunks.

    Uses LLM extraction if llm_client is provided, otherwise falls back
    to heuristic extraction.

    Args:
        chunks: List of evidence chunks (typically top-k after reranking).
        llm_client: Optional LLMClient instance. If None, uses heuristics.

    Returns:
        List of StructuredClaim objects, one per chunk.
    """
    claims = []
    method = "LLM" if llm_client else "heuristic"
    logger.info(f"Extracting claims from {len(chunks)} chunks using {method} method.")

    for chunk in chunks:
        if llm_client:
            claim = extract_claim_with_llm(chunk, llm_client)
        else:
            claim = extract_claim_heuristic(chunk)
        claims.append(claim)

    return claims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_field(data: dict, key: str) -> str:
    val = data.get(key, "Not reported")
    if not val or str(val).strip().lower() in ("", "none", "null", "n/a", "not applicable"):
        return "Not reported"
    return str(val).strip()


def _safe_direction(val: str) -> str:
    valid_map = {
        "reduction": "Reduction",
        "increase": "Increase",
        "no significant change": "No significant change",
        "no significant": "No significant change",
        "unclear": "Unclear",
    }
    return valid_map.get(str(val).strip().lower(), "Unclear")


def _fallback_claim(chunk: Chunk) -> StructuredClaim:
    """Return a minimal claim with raw text when extraction fails."""
    return StructuredClaim(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        section=chunk.section,
        raw_text=chunk.text,
    )
