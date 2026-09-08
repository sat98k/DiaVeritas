# =============================================================================
# src/preprocessing/entity_extractor.py
#
# Biomedical named entity recognition using scispaCy.
#
# Extracts biomedical entities from chunk text and adds them to chunk metadata.
# This enriches the evidence for:
#   - Structured claim extraction (Phase 3)
#   - Contextual comparison (Phase 3)
#   - Query normalization (Phase 3)
#
# Entity categories we track:
#   - disease: T2D, diabetes, cardiovascular disease, etc.
#   - drug/intervention: metformin, SGLT2 inhibitors, GLP-1 agonists, etc.
#   - chemical/biomarker: HbA1c, glucose, insulin, creatinine, etc.
#   - population: inferred from context (not a standard NER category)
#   - outcome: clinical outcomes (not a standard NER category)
#
# scispaCy en_core_sci_sm gives us ENTITY spans with broad biomedical types.
# We then categorize them using keyword matching for the categories above.
#
# ARCHITECTURE NOTE:
# This module is designed so the entity extraction approach can be replaced
# (e.g., with a BERT-based biomedical NER) without changing the interface.
# The output is always: Dict[str, List[str]] with category keys.
#
# IMPORTANT:
# If scispaCy is unavailable, this module falls back to a keyword-matching
# approach. The failure is logged clearly — not silently ignored.
# =============================================================================

from __future__ import annotations

import re
from typing import Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# scispaCy model loading (lazy, so import errors are deferred to use time)
# ---------------------------------------------------------------------------

_nlp = None
_SCISPACY_AVAILABLE = False
_MODEL_TRIED = False


def _load_model():
    """
    Lazy-load the scispaCy model on first use.

    Uses en_core_sci_sm (small model, ~100MB).
    Falls back to basic keyword extraction if unavailable.
    """
    global _nlp, _SCISPACY_AVAILABLE, _MODEL_TRIED

    if _MODEL_TRIED:
        return
    _MODEL_TRIED = True

    try:
        import spacy
        _nlp = spacy.load("en_core_sci_sm")
        _SCISPACY_AVAILABLE = True
        logger.info("scispaCy model 'en_core_sci_sm' loaded successfully.")
    except OSError:
        logger.warning(
            "scispaCy model 'en_core_sci_sm' not found. "
            "Install with: py -m pip install https://s3-us-west-2.amazonaws.com/"
            "ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz\n"
            "Falling back to keyword-based entity extraction."
        )
        _SCISPACY_AVAILABLE = False
    except ImportError:
        logger.warning(
            "spaCy not installed. Falling back to keyword-based entity extraction."
        )
        _SCISPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Domain keyword lists for entity categorization
# ---------------------------------------------------------------------------

_DRUG_KEYWORDS = {
    # Biguanides
    "metformin", "glucophage",
    # SGLT2 inhibitors
    "empagliflozin", "jardiance", "dapagliflozin", "farxiga", "canagliflozin",
    "invokana", "ertugliflozin", "steglatro", "sotagliflozin", "sglt2",
    "sglt-2", "sglt2 inhibitor",
    # GLP-1 receptor agonists
    "semaglutide", "ozempic", "wegovy", "rybelsus", "liraglutide", "victoza",
    "saxenda", "dulaglutide", "trulicity", "exenatide", "byetta", "bydureon",
    "albiglutide", "glp-1", "glp1", "glp-1 agonist", "glp-1 receptor agonist",
    # DPP-4 inhibitors
    "sitagliptin", "januvia", "saxagliptin", "onglyza", "alogliptin", "nesina",
    "linagliptin", "tradjenta", "vildagliptin", "galvus", "dpp-4", "dpp4",
    "gliptin",
    # Sulfonylureas
    "glipizide", "glucotrol", "glyburide", "glibenclamide", "glimepiride",
    "amaryl", "gliclazide", "diamicron", "sulfonylurea", "sulfonylureas",
    # Thiazolidinediones
    "pioglitazone", "actos", "rosiglitazone", "avandia", "thiazolidinedione",
    "tzd",
    # Insulins
    "insulin", "insulin glargine", "lantus", "insulin detemir", "levemir",
    "insulin lispro", "humalog", "insulin aspart", "novolog", "basal insulin",
    "bolus insulin",
    # Combination / other
    "acarbose", "precose", "repaglinide", "nateglinide",
}

_DISEASE_KEYWORDS = {
    "type 2 diabetes", "t2dm", "t2d", "type ii diabetes", "diabetes mellitus",
    "diabetes", "hyperglycemia", "hypoglycemia", "insulin resistance",
    "cardiovascular disease", "cvd", "heart failure", "hf", "myocardial infarction",
    "mi", "stroke", "ckd", "chronic kidney disease", "diabetic nephropathy",
    "diabetic retinopathy", "diabetic neuropathy", "obesity", "metabolic syndrome",
    "hypertension", "dyslipidemia",
}

_BIOMARKER_KEYWORDS = {
    "hba1c", "a1c", "hemoglobin a1c", "glycated hemoglobin", "fasting glucose",
    "fasting blood glucose", "fbg", "postprandial glucose", "ppg", "blood glucose",
    "creatinine", "egfr", "estimated glomerular filtration rate", "uacr",
    "albuminuria", "microalbuminuria", "ldl", "hdl", "triglycerides",
    "cholesterol", "bmi", "body mass index", "weight", "blood pressure",
    "systolic", "diastolic", "bnp", "troponin", "c-reactive protein", "crp",
    "insulin level", "c-peptide", "glucagon",
}

_OUTCOME_KEYWORDS = {
    "cardiovascular event", "cardiovascular outcome", "mace", "major adverse cardiovascular",
    "mortality", "death", "all-cause mortality", "cv death", "hospitalization",
    "hospital admission", "heart failure hospitalization", "hf hospitalization",
    "kidney outcome", "renal outcome", "progression to esrd", "dialysis",
    "glycemic control", "glycaemic control", "weight loss", "weight reduction",
    "blood pressure reduction", "hypoglycemic event", "hypoglycaemia", "adverse event",
    "adverse effect", "safety", "tolerability", "quality of life",
}


def _keyword_match(text_lower: str, keyword_set) -> List[str]:
    """Find all keywords from a set that appear in the text."""
    found = []
    for kw in keyword_set:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            found.append(kw)
    return found


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract biomedical entities from text.

    Returns a dict with category keys:
    {
        "drugs": [...],
        "diseases": [...],
        "biomarkers": [...],
        "outcomes": [...],
        "other_biomedical": [...]  # scispaCy entities not in above categories
    }

    Missing categories are empty lists, never absent.
    """
    _load_model()
    text_lower = text.lower()

    # Always run keyword matching
    drugs = _keyword_match(text_lower, _DRUG_KEYWORDS)
    diseases = _keyword_match(text_lower, _DISEASE_KEYWORDS)
    biomarkers = _keyword_match(text_lower, _BIOMARKER_KEYWORDS)
    outcomes = _keyword_match(text_lower, _OUTCOME_KEYWORDS)
    other_biomedical: List[str] = []

    # If scispaCy is available, also extract model entities not captured above
    if _SCISPACY_AVAILABLE and _nlp is not None:
        try:
            # Limit text length to avoid slow processing on very long chunks
            doc = _nlp(text[:3000])
            for ent in doc.ents:
                ent_text_lower = ent.text.lower().strip()
                # Only add if not already captured by keyword matching
                already_captured = (
                    ent_text_lower in {d.lower() for d in drugs}
                    or ent_text_lower in {d.lower() for d in diseases}
                    or ent_text_lower in {d.lower() for d in biomarkers}
                    or ent_text_lower in {d.lower() for d in outcomes}
                )
                if not already_captured and len(ent_text_lower) > 2:
                    other_biomedical.append(ent.text)
        except Exception as e:
            logger.warning(f"scispaCy entity extraction failed: {e}")

    # Deduplicate while preserving order
    def dedup(lst: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in lst:
            if item.lower() not in seen:
                seen.add(item.lower())
                out.append(item)
        return out

    return {
        "drugs": dedup(drugs),
        "diseases": dedup(diseases),
        "biomarkers": dedup(biomarkers),
        "outcomes": dedup(outcomes),
        "other_biomedical": dedup(other_biomedical[:20]),  # cap to avoid noise
    }


def enrich_chunks_with_entities(chunks) -> None:
    """
    Mutate a list of Chunk objects in-place, adding entity extraction results.

    This is called as the last step of the ingestion pipeline, after chunking.
    """
    logger.info(f"Extracting entities from {len(chunks)} chunks...")
    for chunk in chunks:
        chunk.entities = extract_entities(chunk.text)
    logger.info("Entity extraction complete.")
