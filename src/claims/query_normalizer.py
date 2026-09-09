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
    "improve": "clinical improvement",
    "improvement": "clinical improvement",
    "better": "clinical improvement",
    "worsen": "disease progression",
    "progression": "disease progression",
    "remission": "disease remission",
    "control": "glycemic control (HbA1c/glucose)",
    "glycemic control": "glycemic control (HbA1c/glucose)",
}

_LIFESTYLE_KEYWORDS = {
    "exercise": "exercise/physical activity",
    "exercising": "exercise/physical activity",
    "exercises": "exercise/physical activity",
    "physical activity": "exercise/physical activity",
    "physically active": "exercise/physical activity",
    "walking": "exercise/physical activity",
    "walk": "exercise/physical activity",
    "aerobic": "aerobic exercise",
    "resistance training": "resistance training",
    "strength training": "resistance training",
    "yoga": "yoga",
    "diet": "dietary intervention",
    "dietary": "dietary intervention",
    "dieting": "dietary intervention",
    "nutrition": "dietary intervention",
    "fasting": "fasting/intermittent fasting",
    "intermittent fasting": "fasting/intermittent fasting",
    "lifestyle": "lifestyle intervention",
    "lifestyle modification": "lifestyle intervention",
    "weight loss program": "weight management",
    "weight loss": "weight management",
    "bariatric surgery": "bariatric surgery",
    "meditation": "stress management",
    "sleep": "sleep intervention",
    "smoking cessation": "smoking cessation",
}

_STOPWORDS = {
    "does", "do", "is", "are", "was", "were", "can", "could", "should",
    "would", "will", "the", "a", "an", "in", "on", "for", "with", "of",
    "to", "and", "or", "by", "from", "at", "as", "it", "its", "this",
    "that", "these", "those", "be", "been", "being", "have", "has", "had",
    "not", "no", "than", "more", "less", "most", "like", "very", "also",
    "how", "what", "which", "who", "when", "where", "why", "if",
    "about", "into", "through", "between", "after", "before",
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
# Biomedical Drug Classes and Members for Query-Time Concept Expansion
# ---------------------------------------------------------------------------

DRUG_CLASSES: Dict[str, Dict[str, List[str]]] = {
    "sglt2": {
        "class_terms": [
            "sglt2", "sglt-2", "sglt2 inhibitor", "sglt2 inhibitors",
            "sglt-2 inhibitor", "sglt-2 inhibitors",
            "sodium-glucose cotransporter-2 inhibitor",
            "sodium-glucose cotransporter 2 inhibitor",
            "sodium-glucose cotransporter 2",
            "sodium glucose cotransporter 2",
        ],
        "members": [
            "empagliflozin", "jardiance",
            "dapagliflozin", "farxiga", "forxiga",
            "canagliflozin", "invokana",
            "ertugliflozin", "steglatro",
            "sotagliflozin", "ziquenza",
        ],
    },
    "glp1": {
        "class_terms": [
            "glp-1", "glp1", "glp-1 receptor agonist", "glp-1 receptor agonists",
            "glp-1 ra", "glp-1 ras", "glp1-ra", "glp1 ra", "glp1 receptor agonist",
            "glucagon-like peptide-1 receptor agonist", "glp-1 agonist", "glp-1 agonists",
            "incretin mimetic", "incretin mimetics",
        ],
        "members": [
            "semaglutide", "ozempic", "wegovy", "rybelsus",
            "liraglutide", "victoza", "saxenda",
            "dulaglutide", "trulicity",
            "tirzepatide", "mounjaro", "zepbound",
            "exenatide", "byetta", "bydureon",
            "lixisenatide", "adlyxin",
            "albiglutide", "tanzeum",
        ],
    },
    "dpp4": {
        "class_terms": [
            "dpp-4", "dpp4", "dpp-4 inhibitor", "dpp-4 inhibitors",
            "dpp4 inhibitor", "dpp4 inhibitors",
            "dipeptidyl peptidase-4 inhibitor", "dipeptidyl peptidase 4 inhibitor",
            "gliptin", "gliptins",
        ],
        "members": [
            "sitagliptin", "januvia",
            "saxagliptin", "onglyza",
            "linagliptin", "tradjenta",
            "alogliptin", "nesina",
            "vildagliptin", "galvus",
        ],
    },
    "sulfonylurea": {
        "class_terms": [
            "sulfonylurea", "sulfonylureas", "su",
        ],
        "members": [
            "glimepiride", "amaryl",
            "glipizide", "glucotrol",
            "glyburide", "glibenclamide", "diabeta", "micronase",
            "gliclazide", "diamicron",
        ],
    },
    "tzd": {
        "class_terms": [
            "tzd", "tzds", "thiazolidinedione", "thiazolidinediones", "glitazone", "glitazones",
        ],
        "members": [
            "pioglitazone", "actos",
            "rosiglitazone", "avandia",
        ],
    },
    "biguanide": {
        "class_terms": [
            "biguanide", "biguanides",
        ],
        "members": [
            "metformin", "glucophage", "fortamet", "glumetza",
        ],
    },
}

OUTCOME_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "heart_failure": [
        "heart failure", "hospitalization for heart failure", "hf hospitalization",
        "hfh", "hhf", "hospital admission for heart failure", "chf", "congestive heart failure",
        "hospitalization", "heart failure events",
    ],
    "cv_events": [
        "cardiovascular events/risk", "cardiovascular events", "cardiovascular risk",
        "cardiovascular", "cardiovascular mortality", "mace", "major adverse cardiovascular",
        "major adverse cardiovascular events", "cardiovascular death", "cv death",
        "myocardial infarction", "heart attack", "stroke", "cardiovascular outcomes",
        "cardiovascular disease", "cvd", "coronary heart disease",
    ],
    "glycemic_control": [
        "glycemic control (hba1c/glucose)", "glycemic control", "glycaemic control",
        "hba1c", "a1c", "blood glucose", "fasting plasma glucose", "fpg",
        "fasting blood glucose", "hyperglycemia", "blood sugar",
    ],
    "renal_outcomes": [
        "renal outcomes", "kidney disease", "nephropathy", "diabetic kidney disease",
        "ckd", "chronic kidney disease", "egfr", "estimated glomerular filtration rate",
        "albuminuria", "microalbuminuria", "macroalbuminuria", "end-stage renal disease", "esrd",
        "progression of kidney disease", "renal", "kidney",
    ],
    "mortality": [
        "mortality", "all-cause mortality", "death", "overall survival", "fatal outcomes", "survival",
    ],
    "weight": [
        "weight/bmi", "weight loss", "weight reduction", "body weight", "bmi", "body mass index", "adiposity",
    ],
    "hypoglycemia": [
        "hypoglycemia", "hypoglycaemia", "hypoglycemic events", "low blood sugar",
    ],
}

DRUG_BRAND_MAP: Dict[str, List[str]] = {
    "empagliflozin": ["jardiance"],
    "jardiance": ["empagliflozin"],
    "dapagliflozin": ["farxiga", "forxiga"],
    "farxiga": ["dapagliflozin"],
    "forxiga": ["dapagliflozin"],
    "canagliflozin": ["invokana"],
    "invokana": ["canagliflozin"],
    "ertugliflozin": ["steglatro"],
    "steglatro": ["ertugliflozin"],
    "semaglutide": ["ozempic", "wegovy", "rybelsus"],
    "ozempic": ["semaglutide"],
    "wegovy": ["semaglutide"],
    "rybelsus": ["semaglutide"],
    "liraglutide": ["victoza", "saxenda"],
    "victoza": ["liraglutide"],
    "saxenda": ["liraglutide"],
    "dulaglutide": ["trulicity"],
    "trulicity": ["dulaglutide"],
    "tirzepatide": ["mounjaro", "zepbound"],
    "mounjaro": ["tirzepatide"],
    "sitagliptin": ["januvia"],
    "januvia": ["sitagliptin"],
    "metformin": ["glucophage"],
    "glucophage": ["metformin"],
    "pioglitazone": ["actos"],
    "actos": ["pioglitazone"],
}


def expand_interventions(interventions: List[str], query_text: str = "") -> List[str]:
    """
    Expand interventions using class-to-member or member-to-class mappings.
    - If a drug class is queried (e.g., 'SGLT2 inhibitors'), expands to all member drugs
      and brand names so member-level trials (e.g. empagliflozin in EMPA-REG) are admitted.
    - If a specific drug is queried (e.g., 'empagliflozin'), expands to its brand names
      and class name, but NOT sister drugs (e.g., dapagliflozin).
    """
    expanded = set()
    for item in interventions:
        expanded.add(item.lower())

    combined_check = (query_text.lower() + " " + " ".join(interventions).lower())

    for class_id, info in DRUG_CLASSES.items():
        # Check if the class itself is targeted
        class_hit = any(
            re.search(r"\b" + re.escape(term) + r"\b", combined_check)
            for term in info["class_terms"]
        )
        if class_hit:
            for term in info["class_terms"]:
                expanded.add(term.lower())
            for member in info["members"]:
                expanded.add(member.lower())
            continue

        # Check if a specific member of this class was mentioned
        for member in info["members"]:
            if any(re.search(r"\b" + re.escape(member) + r"\b", int_item) for int_item in list(expanded)):
                for class_term in info["class_terms"][:3]:  # canonical class terms
                    expanded.add(class_term.lower())

    # Map member generics to brand names and vice versa
    for drug_name, brands in DRUG_BRAND_MAP.items():
        if any(re.search(r"\b" + re.escape(drug_name) + r"\b", int_item) for int_item in list(expanded)):
            expanded.add(drug_name)
            for b in brands:
                expanded.add(b)

    return sorted(list(expanded))


def expand_outcomes(outcomes: List[str], query_text: str = "") -> List[str]:
    """
    Expand outcomes with clinical synonyms and abbreviations.
    """
    expanded = set()
    for item in outcomes:
        expanded.add(item.lower())

    combined_check = (query_text.lower() + " " + " ".join(outcomes).lower())

    for cluster_id, synonyms in OUTCOME_SYNONYM_CLUSTERS.items():
        cluster_hit = any(
            re.search(r"\b" + re.escape(syn) + r"\b", combined_check)
            for syn in synonyms
        )
        if cluster_hit:
            for syn in synonyms:
                expanded.add(syn.lower())

    return sorted(list(expanded))


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
    - expanded_interventions: List[str] — class members and drug aliases for concept gating
    - expanded_outcomes: List[str]      — clinical synonyms for concept gating
    - normalized_text: str     — search-optimized query string for retrieval
    """
    query_lower = query.lower()

    # Detect diseases
    diseases = _keyword_match(query_lower, _DISEASE_KEYWORDS)
    # Always include T2D if not explicitly matched but implied
    if not diseases and any(kw in query_lower for kw in ["diabetes", "t2d", "t2dm"]):
        diseases = ["Type 2 Diabetes"]

    # Detect interventions (drugs and lifestyle)
    interventions = _keyword_match(query_lower, _DRUG_KEYWORDS)
    for kw, label in _LIFESTYLE_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", query_lower) and label not in interventions:
            interventions.append(label)

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

    # Build clean lexical search query directly from user query tokens
    query_tokens = re.findall(r'[a-zA-Z0-9]+', query_lower)
    content_tokens = [w for w in query_tokens if w not in _STOPWORDS and len(w) > 1]

    seen_tokens = set()
    deduped_tokens = []
    for t in content_tokens:
        if t not in seen_tokens:
            seen_tokens.add(t)
            deduped_tokens.append(t)

    # Append high-value domain expansion terms for BM25 retrieval without over-expanding
    expansion_terms = []
    for int_term in interventions:
        int_low = int_term.lower()
        if any(k in int_low for k in ["sglt2", "sglt-2", "sglt 2"]):
            expansion_terms.extend(["empagliflozin", "dapagliflozin", "canagliflozin"])
        elif any(k in int_low for k in ["glp-1", "glp1", "glp 1"]):
            expansion_terms.extend(["semaglutide", "liraglutide", "dulaglutide"])
        elif any(k in int_low for k in ["dpp-4", "dpp4", "dpp 4", "gliptin"]):
            expansion_terms.extend(["sitagliptin", "linagliptin", "saxagliptin"])
        elif any(k in int_low for k in ["tzd", "thiazolidinedione"]):
            expansion_terms.extend(["pioglitazone", "rosiglitazone"])
        elif any(k in int_low for k in ["sulfonylurea"]):
            expansion_terms.extend(["glimepiride", "glipizide", "glyburide"])

    for out_term in normalized_outcomes:
        out_low = out_term.lower()
        if "heart failure" in out_low or "hospitalization" in out_low:
            expansion_terms.extend(["hhf"])
        if "cardiovascular" in out_low or "mace" in out_low:
            expansion_terms.extend(["mace"])

    for exp_t in expansion_terms:
        if exp_t not in seen_tokens:
            seen_tokens.add(exp_t)
            deduped_tokens.append(exp_t)

    query_type = classify_query_intent(query)
    normalized_text = " ".join(deduped_tokens) if deduped_tokens else query
    hypothesis_text = query_to_hypothesis(query)

    expanded_interventions = expand_interventions(interventions, query)
    expanded_outcomes = expand_outcomes(normalized_outcomes, query)

    result = {
        "original_query": query,
        "query_type": query_type,
        "disease": diseases,
        "interventions": interventions,
        "outcomes": normalized_outcomes,
        "biomarkers": biomarkers,
        "population": population,
        "comparator": comparator,
        "expanded_interventions": expanded_interventions,
        "expanded_outcomes": expanded_outcomes,
        "normalized_text": normalized_text,
        "dense_query": query,
        "hypothesis_text": hypothesis_text,
    }

    logger.debug(f"Query normalized ({query_type}): {result}")
    return result


QUERY_TYPE_VERIFICATION = "VERIFICATION"
QUERY_TYPE_DESCRIPTIVE_QA = "DESCRIPTIVE_QA"


def classify_query_intent(query: str) -> str:
    """
    Classify whether a clinical query is:
    - VERIFICATION: hypothesis testing (SUPPORTED / REFUTED / INCONCLUSIVE verdict)
    - DESCRIPTIVE_QA: open informational clinical Q&A (what, how, why, explain, side effects)
    """
    q = query.strip().lower()

    descriptive_triggers = [
        r"^what\s+(is|are|were|was|can|do|does)\b",
        r"^how\s+(does|do|can|to|is|are)\b",
        r"^why\s+(is|are|does|do)\b",
        r"^explain\b",
        r"^describe\b",
        r"^compare\b",
        r"^list\b",
        r"\bmechanism of action\b",
        r"\bguidelines for\b",
        r"\bside effects of\b",
        r"\bdosage of\b",
        r"\boverview of\b",
    ]
    for pattern in descriptive_triggers:
        if re.search(pattern, q):
            # If asking direct head-to-head comparison with comparative direction, consider verification
            if not any(k in q for k in ["more effective than", "better than", "superior to"]):
                return QUERY_TYPE_DESCRIPTIVE_QA

    return QUERY_TYPE_VERIFICATION


def query_to_hypothesis(query: str) -> str:
    """
    Convert an interrogative clinical query into a declarative hypothesis assertion for NLI.

    Examples:
        'Does metformin reduce cardiovascular risk in patients with type 2 diabetes?'
        -> 'Metformin reduces cardiovascular risk in patients with type 2 diabetes.'

        'Is metformin more effective than lifestyle changes for prediabetes?'
        -> 'Metformin is more effective than lifestyle changes for prediabetes.'

        'Are SGLT2 inhibitors safe for patients with chronic kidney disease and T2D?'
        -> 'SGLT2 inhibitors are safe for patients with chronic kidney disease and T2D.'

        'Do GLP-1 receptor agonists improve HbA1c in type 2 diabetes patients?'
        -> 'GLP-1 receptor agonists improve HbA1c in type 2 diabetes patients.'
    """
    q = query.strip()
    q_no_q = q.rstrip("?").strip()
    # Strip parenthetical examples (e.g., "(like walking)", "(e.g., metformin)")
    q_no_q = re.sub(r"\s*\([^)]*\)", "", q_no_q).strip()

    # Pattern: Does X improve with Y? -> Y improves X.
    m_improve = re.match(r"^does\s+(.+?)\s+improve\s+with\s+(.+)$", q_no_q, re.IGNORECASE)
    if m_improve:
        condition, intervention = m_improve.groups()
        return f"{intervention[0].upper() + intervention[1:]} improves {condition}."

    # Pattern: Does / Do X [causative_verb] Y? (Prioritize common clinical verbs so multi-word subjects like 'drinking more coffee' parse properly)
    causative_verbs = r"(increase|decrease|reduce|lower|improve|cause|worsen|prevent|elevate|affect|help|lead to|have)"
    m_cause = re.match(rf"^(does|do)\s+(.+?)\s+{causative_verbs}\s+(.+)$", q_no_q, re.IGNORECASE)
    if m_cause:
        aux, subj, verb, rest = m_cause.groups()
        v = verb.lower()
        if aux.lower() == "does":
            if v == "have":
                v3 = "has"
            elif v == "lead to":
                v3 = "leads to"
            elif v.endswith(("s", "sh", "ch", "x", "z", "o")):
                v3 = v + "es"
            elif v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
                v3 = v[:-1] + "ies"
            else:
                v3 = v + "s"
        else:
            v3 = v
        return f"{subj[0].upper() + subj[1:]} {v3} {rest}."

    # Pattern: Does X [verb] Y? -> X [verbs] Y.
    m = re.match(r"^does\s+(.+?)\s+([a-z]+)\s+(.+)$", q_no_q, re.IGNORECASE)
    if m:
        subj, verb, rest = m.groups()
        v = verb.lower()
        if v == "have":
            v3 = "has"
        elif v.endswith(("s", "sh", "ch", "x", "z", "o")):
            v3 = v + "es"
        elif v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
            v3 = v[:-1] + "ies"
        else:
            v3 = v + "s"
        return f"{subj[0].upper() + subj[1:]} {v3} {rest}."

    # Pattern: Do X [verb] Y? -> X [verb] Y.
    m = re.match(r"^do\s+(.+?)\s+([a-z]+)\s+(.+)$", q_no_q, re.IGNORECASE)
    if m:
        subj, verb, rest = m.groups()
        return f"{subj[0].upper() + subj[1:]} {verb.lower()} {rest}."

    # Pattern: Is / Are / Was / Were X effective in [verb]ing Y? -> X [verb] Y.
    m_eff = re.match(
        r"^(is|are|was|were)\s+([A-Za-z0-9\-\s]+?)\s+effective\s+in\s+([a-z]+)ing\s+(.+)$",
        q_no_q,
        re.IGNORECASE,
    )
    if m_eff:
        copula, subj, verb_stem, rest = m_eff.groups()
        verb = verb_stem.lower()
        if verb == "reduc":
            verb = "reduce"
        elif verb == "improv":
            verb = "improve"
        elif verb == "lower":
            verb = "lower"
        elif verb == "prevent":
            verb = "prevent"
        elif verb == "increas":
            verb = "increase"
        v_final = verb if subj.rstrip().endswith("s") else (verb + "s")
        return f"{subj[0].upper() + subj[1:]} {v_final} {rest}."

    # Pattern: Is / Are / Was / Were X [adjective/predicate] Y? -> X is/are [adjective/predicate] Y.
    m = re.match(
        r"^(is|are|was|were)\s+([A-Za-z0-9\-\s]+?)\s+(more|less|better|superior|inferior|effective|associated|safe|recommended|beneficial|linked|protective|preferred|helpful)\s+(.+)$",
        q_no_q,
        re.IGNORECASE,
    )
    if m:
        copula, subj, predicate, rest = m.groups()
        return f"{subj[0].upper() + subj[1:]} {copula.lower()} {predicate} {rest}."

    # Pattern: Can / Could / Should X [verb] Y? -> X can/could/should [verb] Y.
    m = re.match(r"^(can|could|should)\s+(.+?)\s+(.+)$", q_no_q, re.IGNORECASE)
    if m:
        modal, subj, rest = m.groups()
        return f"{subj[0].upper() + subj[1:]} {modal.lower()} {rest}."

    # Fallback: capitalize first letter and add period
    return f"{q_no_q[0].upper() + q_no_q[1:]}."
