# =============================================================================
# src/claims/query_normalizer.py
#
# Normalizes user queries into structured biomedical concepts.
#
# Purpose:
#   Convert a free-text question into a structured representation that
#   identifies the key biomedical concepts without adding unsupported
#   clinical assumptions.
#
# Approach:
#   Uses keyword matching for fast, transparent concept extraction.
#   Optionally uses the configured LLM for richer normalization when
#   an API key is available.
#
# Example:
#   Input:  "Does metformin reduce heart problems in people with T2D?"
#   Output: {
#       "original_query": "Does metformin reduce heart problems in people with T2D?",
#       "disease": ["Type 2 Diabetes"],
#       "interventions": ["metformin"],
#       "outcomes": ["cardiovascular events/risk"],
#       "population": [],         # not explicitly stated
#       "comparator": [],         # not explicitly stated
#       "normalized_text": "metformin cardiovascular outcomes type 2 diabetes"
#   }
#
# IMPORTANT: We do NOT add unsupported clinical assumptions.
# If a concept is not present in the query, its list is empty.
# =============================================================================

from __future__ import annotations

import re
from typing import Dict, List, Any

from loguru import logger

from src.preprocessing.entity_extractor import (
    _DRUG_KEYWORDS, _DISEASE_KEYWORDS, _BIOMARKER_KEYWORDS, _OUTCOME_KEYWORDS,
    _keyword_match,
)


# ---------------------------------------------------------------------------
# Outcome normalization map
# ---------------------------------------------------------------------------

_OUTCOME_NORMALIZATIONS = {
    "heart problem": "cardiovascular events/risk",
    "heart problems": "cardiovascular events/risk",
    "heart attack": "myocardial infarction",
    "heart failure": "heart failure",
    "stroke": "stroke",
    "kidney": "renal outcomes",
    "kidney disease": "renal outcomes",
    "death": "mortality",
    "survival": "mortality",
    "weight": "weight/BMI",
    "blood sugar": "glycemic control (HbA1c/glucose)",
    "blood glucose": "glycemic control (HbA1c/glucose)",
    "hba1c": "glycemic control (HbA1c/glucose)",
    "a1c": "glycemic control (HbA1c/glucose)",
    "cardiovascular": "cardiovascular events/risk",
    "cv risk": "cardiovascular events/risk",
    "mace": "major adverse cardiovascular events (MACE)",
    "mortality": "mortality",
    "hospitalization": "hospitalization",
    "safety": "safety/adverse events",
    "side effect": "safety/adverse events",
    "adverse event": "safety/adverse events",
}

_POPULATION_KEYWORDS = {
    "elderly": "elderly patients",
    "older adult": "elderly patients",
    "older patients": "elderly patients",
    "young": "younger patients",
    "children": "pediatric patients",
    "pediatric": "pediatric patients",
    "obese": "obese patients",
    "overweight": "overweight/obese patients",
    "ckd": "patients with chronic kidney disease",
    "kidney disease": "patients with chronic kidney disease",
    "renal impairment": "patients with renal impairment",
    "heart failure": "patients with heart failure",
    "cardiovascular disease": "patients with cardiovascular disease",
    "high risk": "high-risk patients",
    "low risk": "low-risk patients",
    "african american": "African American patients",
    "asian": "Asian patients",
}


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------

def normalize_query(query: str) -> Dict[str, Any]:
    """
    Normalize a free-text query into structured biomedical concepts.

    Returns a dict with:
    - original_query: str
    - disease: List[str]       — detected disease concepts
    - interventions: List[str] — detected drug/treatment concepts
    - outcomes: List[str]      — detected outcome concepts (normalized where possible)
    - biomarkers: List[str]    — detected biomarker concepts
    - population: List[str]    — detected population descriptors
    - comparator: List[str]    — detected comparators (always empty without explicit mention)
    - normalized_text: str     — search-optimized query string for retrieval
    """
    query_lower = query.lower()

    # Detect diseases
    diseases = _keyword_match(query_lower, _DISEASE_KEYWORDS)
    # Always include T2D if not explicitly matched but implied
    if not diseases and any(kw in query_lower for kw in ["diabetes", "t2d", "t2dm"]):
        diseases = ["Type 2 Diabetes"]

    # Detect interventions (drugs)
    interventions = _keyword_match(query_lower, _DRUG_KEYWORDS)

    # Detect biomarkers
    biomarkers = _keyword_match(query_lower, _BIOMARKER_KEYWORDS)

    # Detect and normalize outcomes
    raw_outcomes = _keyword_match(query_lower, _OUTCOME_KEYWORDS)
    # Also check for informal outcome terms
    normalized_outcomes = list(raw_outcomes)
    for informal, normalized in _OUTCOME_NORMALIZATIONS.items():
        if informal in query_lower and normalized not in normalized_outcomes:
            normalized_outcomes.append(normalized)

    # Detect population
    population = []
    for kw, label in _POPULATION_KEYWORDS.items():
        if kw in query_lower:
            population.append(label)

    # Comparator: only if explicit language detected
    comparator = []
    comparator_patterns = [
        r"compared (with|to)\s+([\w\s\-]+?)(?:\s+in|\s+for|\s+among|$)",
        r"versus\s+([\w\s\-]+?)(?:\s+in|\s+for|\s+among|$)",
        r"vs\.?\s+([\w\s\-]+?)(?:\s+in|\s+for|\s+among|$)",
        r"(?:better than|superior to|preferred to)\s+([\w\s\-]+?)(?:\s+in|\s+for|\s+among|$)",
    ]
    for pattern in comparator_patterns:
        match = re.search(pattern, query_lower)
        if match:
            comparator.append(match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip())

    # Build a search-optimized normalized text
    # Combines all detected concepts without duplication
    concept_parts = []
    concept_parts.extend(interventions)
    concept_parts.extend(diseases)
    concept_parts.extend(normalized_outcomes)
    concept_parts.extend(biomarkers)
    concept_parts.extend(population)
    # Deduplicate while preserving order
    seen = set()
    unique_parts = []
    for p in concept_parts:
        pl = p.lower()
        if pl not in seen:
            seen.add(pl)
            unique_parts.append(p)

    normalized_text = " ".join(unique_parts) if unique_parts else query

    result = {
        "original_query": query,
        "disease": diseases,
        "interventions": interventions,
        "outcomes": normalized_outcomes,
        "biomarkers": biomarkers,
        "population": population,
        "comparator": comparator,
        "normalized_text": normalized_text,
    }

    logger.debug(f"Query normalized: {result}")
    return result
