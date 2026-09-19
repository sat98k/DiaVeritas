# =============================================================================
# src/claims/comparative_gate.py
#
# Comparative Claim Verification & Comparative Evidence Gate.
#
# Prevents false SUPPORTED verdicts on comparative clinical queries
# (e.g. "Which is better for lowering HbA1c: metformin or sulfonylureas?")
# by enforcing that:
#   1. Both comparative entities (or class members) must be present in the evidence.
#   2. Single-agent efficacy (only A or only B) cannot support superiority.
#   3. Combination / add-on regimens (A + B) cannot establish monotherapy superiority.
#   4. Guideline first-line preference reflects overall clinical utility, not monotherapy outcome superiority.
#   5. Adverse-event differences (hypoglycemia/weight) do not support glycemic outcome superiority.
#   6. Equivalence in head-to-head trials refutes a directional superiority claim.
#   7. Non-comparative queries completely bypass this gate, preserving 100% of existing behavior.
# =============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger

from src.preprocessing.chunker import Chunk
from src.claims.claim_extractor import StructuredClaim


@dataclass
class ComparativeQueryInfo:
    """Structured information extracted from a comparative clinical query."""
    is_comparative: bool = False
    comparative_type: str = "NON_COMPARATIVE"  # "DIRECTIONAL_A_SUPERIOR", "DIRECTIONAL_B_SUPERIOR", "OPEN_CHOICE", "NON_COMPARATIVE"
    intervention_a: str = ""
    intervention_b: str = ""
    intervention_a_terms: List[str] = field(default_factory=list)
    intervention_b_terms: List[str] = field(default_factory=list)
    target_outcome: str = ""
    target_outcome_terms: List[str] = field(default_factory=list)
    population: str = ""
    claimed_superior: Optional[str] = None
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_comparative": self.is_comparative,
            "comparative_type": self.comparative_type,
            "intervention_a": self.intervention_a,
            "intervention_b": self.intervention_b,
            "intervention_a_terms": self.intervention_a_terms,
            "intervention_b_terms": self.intervention_b_terms,
            "target_outcome": self.target_outcome,
            "target_outcome_terms": self.target_outcome_terms,
            "population": self.population,
            "claimed_superior": self.claimed_superior,
            "raw_query": self.raw_query,
        }


# ---------------------------------------------------------------------------
# Query Intent Detection for Comparative Questions
# ---------------------------------------------------------------------------

_COMPARATIVE_PATTERNS = [
    # Pattern 1: Which is better for [outcome] in [population]: [A] or [B]?
    (
        r"^which\s+is\s+(?:better|more\s+effective|superior|preferred)\s+(?:for\s+([^\:\?]+?))?(?::|\s+in\s+([^\:\?]+?):?|\s+between|\s+among)\s*([A-Za-z0-9\-\s]+?)\s+(?:or|vs\.?|versus)\s+([A-Za-z0-9\-\s]+?)(?:\s+in\s+([^\:\?]+?))?[\?\.]?$",
        "OPEN_CHOICE",
    ),
    # Pattern 2: Which is better between [A] and [B] for [outcome]?
    (
        r"^which\s+is\s+(?:better|more\s+effective|superior|preferred)\s+(?:between|among)\s+([A-Za-z0-9\-\s]+?)\s+and\s+([A-Za-z0-9\-\s]+?)(?:\s+for\s+([^\:\?]+?))?(?:\s+in\s+([^\:\?]+?))?[\?\.]?$",
        "OPEN_CHOICE_BETWEEN",
    ),
    # Pattern 3: Is/Are [A] more effective than [B] for [outcome]?
    (
        r"^(?:is|are|was|were)\s+([A-Za-z0-9\-\s]+?)\s+(?:more\s+effective|better|superior|preferred)\s+(?:than|to|over)\s+([A-Za-z0-9\-\s]+?)(?:\s+(?:for|in|reducing|lowering)\s+([^\:\?]+?))?(?:\s+in\s+([^\:\?]+?))?[\?\.]?$",
        "DIRECTIONAL",
    ),
    # Pattern 4: Does/Do [A] [reduce/lower/improve] [outcome] more than [B]?
    (
        r"^(?:does|do)\s+([A-Za-z0-9\-\s]+?)\s+(reduce|lower|improve|decrease|increase)\s+(.+?)\s+more\s+than\s+([A-Za-z0-9\-\s]+?)(?:\s+in\s+(.+?))?[\?\.]?$",
        "DIRECTIONAL_VERB",
    ),
    # Pattern 5: [A] vs/versus [B] for [outcome]
    (
        r"^([A-Za-z0-9\-\s]+?)\s+(?:vs\.?|versus)\s+([A-Za-z0-9\-\s]+?)(?:\s+(?:for|in)\s+([^\:\?]+?))?(?:\s+in\s+([^\:\?]+?))?[\?\.]?$",
        "OPEN_VS",
    ),
]


def _clean_entity(s: str) -> str:
    """Strip punctuation and extraneous words from extracted entity names."""
    if not s:
        return ""
    s = s.strip().strip("?:;.,")
    s = re.sub(r"^(?:patients with|patients|individuals with|individuals)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


def detect_comparative_query(query: str, normalized_query: Optional[Dict[str, Any]] = None) -> ComparativeQueryInfo:
    """
    Detect whether a query expresses comparative intent between two interventions.

    Identifies:
      - Intervention A & Intervention B
      - Target outcome
      - Directionality (A > B vs B > A vs Open Choice)
      - Concept expansions for A, B, and Outcome
    """
    q = query.strip()
    q_lower = q.lower()

    # Fast reject if no comparative triggers present
    triggers = ["better", "more effective", "superior", "preferred", "more than", " vs ", " vs. ", " versus ", " or "]
    if not any(t in q_lower for t in triggers):
        return ComparativeQueryInfo(raw_query=query)

    for pattern, ptype in _COMPARATIVE_PATTERNS:
        m = re.search(pattern, q, re.IGNORECASE)
        if not m:
            continue

        groups = m.groups()
        if ptype == "OPEN_CHOICE":
            # (outcome_or_pop1, pop2, int_a, int_b, pop3)
            out_pop1, pop2, int_a, int_b, pop3 = groups
            target_out = out_pop1 or ""
            population = pop2 or pop3 or ""
            int_a = _clean_entity(int_a)
            int_b = _clean_entity(int_b)
            claimed_superior = None
            comp_type = "OPEN_CHOICE"

        elif ptype == "OPEN_CHOICE_BETWEEN":
            int_a, int_b, target_out, population = groups
            int_a = _clean_entity(int_a)
            int_b = _clean_entity(int_b)
            target_out = target_out or ""
            population = population or ""
            claimed_superior = None
            comp_type = "OPEN_CHOICE"

        elif ptype == "DIRECTIONAL":
            int_a, int_b, target_out, population = groups
            int_a = _clean_entity(int_a)
            int_b = _clean_entity(int_b)
            target_out = target_out or ""
            population = population or ""
            claimed_superior = int_a
            comp_type = "DIRECTIONAL_A_SUPERIOR"

        elif ptype == "DIRECTIONAL_VERB":
            int_a, verb, target_out, int_b, population = groups
            int_a = _clean_entity(int_a)
            int_b = _clean_entity(int_b)
            target_out = f"{verb} {target_out}".strip()
            population = population or ""
            claimed_superior = int_a
            comp_type = "DIRECTIONAL_A_SUPERIOR"

        elif ptype == "OPEN_VS":
            int_a, int_b, target_out, population = groups
            int_a = _clean_entity(int_a)
            int_b = _clean_entity(int_b)
            target_out = target_out or ""
            population = population or ""
            claimed_superior = None
            comp_type = "OPEN_CHOICE"
        else:
            continue

        # Basic validity: need distinct non-empty intervention strings
        if not int_a or not int_b or int_a.lower() == int_b.lower():
            continue

        # Derive expansions using query normalizer maps if available
        a_terms = _expand_intervention_entity(int_a)
        b_terms = _expand_intervention_entity(int_b)
        out_terms = _expand_outcome_entity(target_out)

        return ComparativeQueryInfo(
            is_comparative=True,
            comparative_type=comp_type,
            intervention_a=int_a,
            intervention_b=int_b,
            intervention_a_terms=a_terms,
            intervention_b_terms=b_terms,
            target_outcome=target_out,
            target_outcome_terms=out_terms,
            population=population,
            claimed_superior=claimed_superior,
            raw_query=query,
        )

    # Heuristic fallback: if normalized_query has 2 interventions and comparator or 'vs' / 'or'
    if normalized_query and len(normalized_query.get("interventions", [])) >= 2:
        ints = normalized_query["interventions"]
        if any(w in q_lower for w in ["better", "more effective", "superior", " vs ", " vs. ", "versus", " or "]):
            int_a = ints[0]
            int_b = ints[1]
            return ComparativeQueryInfo(
                is_comparative=True,
                comparative_type="OPEN_CHOICE",
                intervention_a=int_a,
                intervention_b=int_b,
                intervention_a_terms=_expand_intervention_entity(int_a),
                intervention_b_terms=_expand_intervention_entity(int_b),
                target_outcome=" ".join(normalized_query.get("outcomes", [])),
                target_outcome_terms=_expand_outcome_entity(" ".join(normalized_query.get("outcomes", []))),
                population=" ".join(normalized_query.get("population", [])),
                claimed_superior=None,
                raw_query=query,
            )

    return ComparativeQueryInfo(raw_query=query)


# ---------------------------------------------------------------------------
# Biomedical Entity Expansions
# ---------------------------------------------------------------------------

_ALIAS_TABLE: Dict[str, List[str]] = {
    "metformin": ["metformin", "glucophage", "biguanide", "biguanides"],
    "sulfonylurea": [
        "sulfonylurea", "sulfonylureas", "glimepiride", "glipizide", "glyburide",
        "gliclazide", "glibenclamide", "amaryl", "glucotrol", "diabeta", "micronase",
        "diamicron", "su",
    ],
    "sulfonylureas": [
        "sulfonylurea", "sulfonylureas", "glimepiride", "glipizide", "glyburide",
        "gliclazide", "glibenclamide", "amaryl", "glucotrol", "diabeta", "micronase",
        "diamicron", "su",
    ],
    "sglt2": [
        "sglt2", "sglt-2", "sglt2 inhibitor", "sglt2 inhibitors", "empagliflozin",
        "dapagliflozin", "canagliflozin", "ertugliflozin", "jardiance", "farxiga", "forxiga", "invokana",
    ],
    "sglt2 inhibitors": [
        "sglt2", "sglt-2", "sglt2 inhibitor", "sglt2 inhibitors", "empagliflozin",
        "dapagliflozin", "canagliflozin", "ertugliflozin", "jardiance", "farxiga", "forxiga", "invokana",
    ],
    "glp1": [
        "glp-1", "glp1", "glp-1 receptor agonist", "glp-1 receptor agonists", "glp-1 ra",
        "semaglutide", "liraglutide", "dulaglutide", "tirzepatide", "ozempic", "wegovy", "rybelsus", "victoza", "trulicity", "mounjaro",
    ],
    "glp-1 receptor agonists": [
        "glp-1", "glp1", "glp-1 receptor agonist", "glp-1 receptor agonists", "glp-1 ra",
        "semaglutide", "liraglutide", "dulaglutide", "tirzepatide", "ozempic", "wegovy", "rybelsus", "victoza", "trulicity", "mounjaro",
    ],
    "lifestyle": ["lifestyle", "lifestyle intervention", "exercise", "diet", "physical activity"],
    "exercise": ["exercise", "physical activity", "resistance training", "aerobic exercise", "walking"],
}

_OUTCOME_ALIAS_TABLE: Dict[str, List[str]] = {
    "hba1c": ["hba1c", "a1c", "glycated hemoglobin", "glycated haemoglobin", "glycemic control", "glycaemic control", "blood glucose"],
    "glycemic control": ["hba1c", "a1c", "glycemic control", "glycaemic control", "blood glucose", "fpg"],
    "cardiovascular mortality": ["cardiovascular mortality", "cv mortality", "cv death", "cardiovascular death"],
    "heart failure": ["heart failure", "hf", "hospitalization for heart failure", "hfh"],
}


def _expand_intervention_entity(ent: str) -> List[str]:
    """Expand drug/intervention string to aliases and class members."""
    ent_clean = ent.lower().strip()
    res = {ent_clean}
    for k, aliases in _ALIAS_TABLE.items():
        if k in ent_clean or any(a in ent_clean for a in aliases):
            res.update(aliases)
    return sorted(list(res))


def _expand_outcome_entity(out: str) -> List[str]:
    """Expand outcome string to clinical synonyms."""
    out_clean = out.lower().strip()
    res = {out_clean}
    for k, aliases in _OUTCOME_ALIAS_TABLE.items():
        if k in out_clean or any(a in out_clean for a in aliases):
            res.update(aliases)
    if any(term in out_clean for term in ["hba1c", "a1c", "glycem", "blood glucose"]):
        res.update(["hba1c", "a1c", "glycated hemoglobin", "glycemic control", "blood glucose"])
    return sorted(list(res))


# ---------------------------------------------------------------------------
# Comparative Evidence Verification Gate
# ---------------------------------------------------------------------------

_COMBINATION_PATTERNS = [
    r"\badd-?on\s+(?:to|therapy|with)\b",
    r"\badded\s+to\b",
    r"\bin\s+combination\s+with\b",
    r"\bcombination\s+therap(?:y|ies)\b",
    r"\bcombined\s+with\b",
    r"\bplus\b",
    r"\bco-?administration\b",
    r"\bdual\s+therapy\b",
    r"\btriple\s+therapy\b",
    r"\bbackground\s+(?:therapy|metformin)\b",
]

_GUIDELINE_PATTERNS = [
    r"\bfirst-?line\s+(?:therapy|pharmacotherapy|agent|treatment|drug|recommendation)\b",
    r"\brecommend(?:ed|s)?\s+(?:metformin\s+as\s+first-?line|as\s+first-?line)\b",
    r"\bpreferred\s+(?:initial|first-?line)\b",
    r"\bguidelines?\s+recommend\b",
]

_EQUIVALENCE_PATTERNS = [
    r"\bsimilar\s+(?:efficacy|reductions?|glyc[ae]mic|hba1c|glucose-lowering|effects?)\b",
    r"\bcomparable\s+(?:efficacy|reductions?|glyc[ae]mic|hba1c|glucose-lowering|effects?)\b",
    r"\bequivalent\s+(?:efficacy|reductions?|glyc[ae]mic|hba1c|glucose-lowering|effects?)\b",
    r"\bno\s+significant\s+difference\s+(?:in|between)\s+(?:hba1c|glyc[ae]mic|glucose|efficacy)\b",
    r"\bno\s+difference\s+(?:in|between)\s+(?:hba1c|glyc[ae]mic|glucose|efficacy)\b",
    r"\bnon-?inferior\b",
]

_ADVERSE_EVENT_PATTERNS = [
    r"\bhypoglyc[ae]mia\b",
    r"\bweight\s+(?:gain|loss|increase|reduction)\b",
    r"\bgastrointestinal\b",
    r"\btolerability\b",
    r"\badverse\s+events?\b",
    r"\bside\s+effects?\b",
]


def evaluate_comparative_chunk(
    chunk: Chunk,
    claim: Optional[StructuredClaim],
    comp_info: ComparativeQueryInfo,
    nli_label: str,
    nli_confidence: float,
) -> Tuple[str, str]:
    """
    Evaluate an evidence chunk against comparative verification criteria.

    Returns:
        (relationship, rationale)
        where relationship is one of: "Supports", "Contradicts", "Contextual Difference", "Neutral"
    """
    if not comp_info.is_comparative:
        # Non-comparative: preserve standard NLI relationship
        if nli_label == "ENTAILMENT" and nli_confidence >= 0.5:
            return "Supports", "Direct entailment"
        elif nli_label == "CONTRADICTION" and nli_confidence >= 0.5:
            return "Contradicts", "Direct contradiction"
        return "Neutral", "Neutral evidence"

    text_lower = chunk.text.lower()
    title_lower = (chunk.title or "").lower()
    combined_text = f"{title_lower} {text_lower}"

    # Criterion 1: Presence of both comparative entities
    has_a = any(re.search(r"\b" + re.escape(t) + r"\b", combined_text) for t in comp_info.intervention_a_terms)
    has_b = any(re.search(r"\b" + re.escape(t) + r"\b", combined_text) for t in comp_info.intervention_b_terms)

    if not (has_a and has_b):
        # Missing one or both interventions -> cannot prove comparative claim
        missing = []
        if not has_a:
            missing.append(comp_info.intervention_a)
        if not has_b:
            missing.append(comp_info.intervention_b)
        return "Neutral", f"Comparative Gate: Single-agent evidence; missing comparative comparator ({', '.join(missing)})"

    # Criterion 2: Combination / Add-on therapy filter
    # If the study evaluates A + B together rather than A vs B head-to-head
    is_combination = any(re.search(p, combined_text) for p in _COMBINATION_PATTERNS)
    # Check if specifically "A and B combination", "A-B combination", "B added to A"
    combo_specific = bool(
        re.search(rf"\b{re.escape(comp_info.intervention_a)}[\-\s]+(?:and|plus|with|\-)\s+{re.escape(comp_info.intervention_b)}\b", combined_text)
        or re.search(rf"\b{re.escape(comp_info.intervention_b)}[\-\s]+(?:and|plus|with|\-)\s+{re.escape(comp_info.intervention_a)}\b", combined_text)
        or re.search(r"\b(combination|add-?on|added to|dual therapy)\b", combined_text)
    )

    # If it's a combination/add-on study and does NOT have explicit monotherapy head-to-head arms:
    has_monotherapy_vs = bool(re.search(r"\b(?:monotherapy|versus|vs\.?|head-to-head|compared with)\b", combined_text))
    if (is_combination or combo_specific) and not ("monotherapy" in combined_text and "versus" in combined_text):
        # A + B combo does not demonstrate monotherapy superiority of A over B
        return "Neutral", "Comparative Gate: Combination or add-on regimen (A+B) does not establish monotherapy comparative superiority"

    # Criterion 3: Guideline / First-line recommendation preference
    is_guideline = any(re.search(p, combined_text) for p in _GUIDELINE_PATTERNS)
    # If it cites guideline first-line recommendation without monotherapy trial data:
    if is_guideline and not any(re.search(p, combined_text) for p in _EQUIVALENCE_PATTERNS):
        # Check if text reports actual trial HbA1c comparative difference
        has_direct_num_diff = bool(re.search(r"\b(?:greater|superior|more|larger)\s+(?:reduction|decrease|lowering)\s+in\s+hba1c\b", text_lower))
        if not has_direct_num_diff:
            return "Neutral", "Comparative Gate: Guideline first-line recommendation reflects overall clinical balance, not isolated glycemic outcome superiority"

    # Criterion 4: Endpoint / Outcome check
    # If query outcome is HbA1c/glycemic control, but the chunk only discusses adverse events (hypo/weight)
    is_glycemic_query = any(k in comp_info.target_outcome.lower() for k in ["hba1c", "a1c", "glycem", "glucose", "lowering"])
    if is_glycemic_query:
        has_glycemic_data = any(k in combined_text for k in ["hba1c", "a1c", "glycated", "glucose"])
        if not has_glycemic_data:
            return "Neutral", "Comparative Gate: Evidence does not report on the target glycemic outcome (HbA1c)"

    # Criterion 5: Head-to-Head Comparative Findings
    # Case A: Equivalence / Comparable efficacy reported
    is_equivalent = any(re.search(p, combined_text) for p in _EQUIVALENCE_PATTERNS)
    if is_equivalent:
        if comp_info.comparative_type in ("DIRECTIONAL_A_SUPERIOR", "DIRECTIONAL_B_SUPERIOR"):
            # Equivalence directly REFUTES the claim that A is superior to B (or B superior to A)
            return "Contradicts", "Comparative Gate: Head-to-head trial reports equivalent glycemic efficacy, refuting superiority claim"
        else:
            # Open choice: equivalence means neither is "better"
            return "Neutral", "Comparative Gate: Head-to-head evidence demonstrates therapeutic equivalence; neither agent is superior"

    # Case B: Direct superiority of Intervention A over B
    # e.g. "metformin reduced HbA1c significantly more than sulfonylureas"
    pat_a_superior = rf"\b{re.escape(comp_info.intervention_a)}\b[^\.\;]*\b(?:greater|more|superior|larger)\b[^\.\;]*\b(?:reduction|decrease|lowering)\b[^\.\;]*\b{re.escape(comp_info.intervention_b)}\b"
    if re.search(pat_a_superior, combined_text):
        if comp_info.comparative_type == "DIRECTIONAL_A_SUPERIOR":
            return "Supports", f"Comparative Gate: Head-to-head evidence shows {comp_info.intervention_a} superior to {comp_info.intervention_b}"
        elif comp_info.comparative_type == "DIRECTIONAL_B_SUPERIOR":
            return "Contradicts", f"Comparative Gate: Head-to-head evidence shows {comp_info.intervention_a} superior, refuting {comp_info.intervention_b}"
        else:
            return "Supports", f"Comparative Gate: Head-to-head evidence shows {comp_info.intervention_a} superior to {comp_info.intervention_b}"

    # Case C: Direct superiority of Intervention B over A
    pat_b_superior = rf"\b{re.escape(comp_info.intervention_b)}\b[^\.\;]*\b(?:greater|more|superior|larger)\b[^\.\;]*\b(?:reduction|decrease|lowering)\b[^\.\;]*\b{re.escape(comp_info.intervention_a)}\b"
    if re.search(pat_b_superior, combined_text):
        if comp_info.comparative_type == "DIRECTIONAL_B_SUPERIOR":
            return "Supports", f"Comparative Gate: Head-to-head evidence shows {comp_info.intervention_b} superior to {comp_info.intervention_a}"
        elif comp_info.comparative_type == "DIRECTIONAL_A_SUPERIOR":
            return "Contradicts", f"Comparative Gate: Head-to-head evidence shows {comp_info.intervention_b} superior, refuting {comp_info.intervention_a}"
        else:
            return "Supports", f"Comparative Gate: Head-to-head evidence shows {comp_info.intervention_b} superior to {comp_info.intervention_a}"

    # Fallback: if evidence contains both entities but lacks direct head-to-head outcome comparison
    return "Neutral", "Comparative Gate: Passage mentions both agents but does not report direct head-to-head comparative outcome data"
