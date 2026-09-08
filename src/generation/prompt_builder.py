# =============================================================================
# src/generation/prompt_builder.py
#
# Assembles the structured prompt for LLM final answer synthesis.
#
# The prompt is carefully designed to:
#   1. Provide the LLM with the pre-analyzed evidence (not raw retrieval)
#   2. Instruct the LLM to synthesize and communicate, not invent
#   3. Request citation-grounded statements
#   4. Require acknowledgment of contradictions and uncertainty
#   5. Prohibit unsupported medical claims
#
# The LLM receives:
#   - Original user question
#   - Normalized query concepts
#   - Evidence status (SUPPORTED / REFUTED / INCONCLUSIVE)
#   - Supporting evidence with citations
#   - Contradicting evidence with citations
#   - Contextual differences
#   - NLI summary statistics
# =============================================================================

from __future__ import annotations

from typing import List, Optional

from src.evidence.relationship import EvidenceRelationshipSummary
from src.evidence.status import EvidenceStatusResult
from src.claims.query_normalizer import normalize_query


# ---------------------------------------------------------------------------
# System prompt (constant)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are DiaVeritas, a research assistant that analyzes biomedical literature evidence about Type 2 Diabetes treatment.

Your role is to SYNTHESIZE and COMMUNICATE pre-analyzed evidence. You do NOT:
- Invent evidence
- Claim medical truth beyond what the retrieved literature states
- Make clinical recommendations
- Ignore contradictions or uncertainty

You MUST:
- Directly answer the question based on the analyzed evidence
- Cite specific papers for key claims using [Author Year, Journal] format
- Acknowledge contradictions, contextual differences, and uncertainty
- Clearly state the evidence status (SUPPORTED / REFUTED / INCONCLUSIVE)
- Use hedged language when evidence is conflicting or limited
- Note if important context differs between studies (population, duration, etc.)

You are NOT a clinical decision support system. Include this note:
"This analysis is for research purposes only. It is not medical advice."
"""


# ---------------------------------------------------------------------------
# Evidence formatter
# ---------------------------------------------------------------------------

def _format_evidence_item(item_dict: dict, index: int) -> str:
    """Format a single evidence item for the prompt."""
    chunk = item_dict.get("chunk", {})
    claim = item_dict.get("claim", {})
    nli = item_dict.get("nli", {})

    title = item_dict.get("title", "Not reported")
    authors = item_dict.get("authors", ["Not reported"])
    year = item_dict.get("year", "Unknown")
    journal = item_dict.get("journal", "Not reported")
    section = item_dict.get("section", "Unknown")
    study_type = item_dict.get("study_type", "Not reported")
    text = item_dict.get("text", "")[:500]  # Cap for prompt length
    relationship = item_dict.get("relationship", "Neutral")

    # Citation string
    first_author = authors[0].split(",")[0] if authors else "Unknown"
    citation = f"[{first_author} {year}, {journal}]"

    # Claim summary
    intervention = claim.get("intervention", "Not reported")
    outcome = claim.get("outcome", "Not reported")
    direction = claim.get("direction", "Unclear")
    magnitude = claim.get("magnitude", "Not reported")
    duration = claim.get("duration", "Not reported")

    lines = [
        f"[Evidence {index}] {citation}",
        f"  Relationship: {relationship}",
        f"  Study Type: {study_type} | Section: {section}",
        f"  Claim: {intervention} → {direction} → {outcome}",
    ]
    if magnitude != "Not reported":
        lines.append(f"  Magnitude: {magnitude}")
    if duration != "Not reported":
        lines.append(f"  Duration: {duration}")
    lines.append(f"  Passage: \"{text}...\"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    question: str,
    normalized_query: dict,
    evidence_summary: EvidenceRelationshipSummary,
    status_result: EvidenceStatusResult,
    max_evidence_per_group: int = 4,
) -> str:
    """
    Build the full LLM synthesis prompt.

    Args:
        question: Original user question.
        normalized_query: Output of query_normalizer.normalize_query().
        evidence_summary: EvidenceRelationshipSummary from relationship.py.
        status_result: EvidenceStatusResult from status.py.
        max_evidence_per_group: Max evidence items per category in the prompt.

    Returns:
        Complete prompt string ready for LLMClient.complete().
    """
    prompt_parts = []

    # --- Header ---
    prompt_parts.append("=" * 60)
    prompt_parts.append("EVIDENCE ANALYSIS FOR TYPE 2 DIABETES RESEARCH")
    prompt_parts.append("=" * 60)

    # --- Question ---
    prompt_parts.append(f"\nQUESTION:\n{question}")

    # --- Normalized concepts ---
    interventions = normalized_query.get("interventions", [])
    outcomes = normalized_query.get("outcomes", [])
    population = normalized_query.get("population", [])

    prompt_parts.append("\nKEY CONCEPTS IDENTIFIED:")
    if interventions:
        prompt_parts.append(f"  Interventions: {', '.join(interventions)}")
    if outcomes:
        prompt_parts.append(f"  Outcomes: {', '.join(outcomes)}")
    if population:
        prompt_parts.append(f"  Population: {', '.join(population)}")

    # --- Evidence status ---
    prompt_parts.append(f"\nEVIDENCE STATUS: {status_result.status}")
    prompt_parts.append(f"Evidence count: {status_result.n_total} items analyzed")
    prompt_parts.append(
        f"  Supporting: {status_result.n_supporting} | "
        f"Contradicting: {status_result.n_contradicting} | "
        f"Contextual: {status_result.n_contextual} | "
        f"Neutral: {status_result.n_neutral}"
    )
    prompt_parts.append(f"Status rationale: {status_result.rationale}")

    # --- Supporting evidence ---
    if evidence_summary.supporting:
        prompt_parts.append("\nSUPPORTING EVIDENCE:")
        for i, item in enumerate(evidence_summary.supporting[:max_evidence_per_group], 1):
            prompt_parts.append(_format_evidence_item(item.to_dict(), i))

    # --- Contradicting evidence ---
    if evidence_summary.contradicting:
        prompt_parts.append("\nCONTRADICTING EVIDENCE:")
        for i, item in enumerate(evidence_summary.contradicting[:max_evidence_per_group], 1):
            prompt_parts.append(_format_evidence_item(item.to_dict(), i))

    # --- Contextual differences ---
    if evidence_summary.contextual:
        prompt_parts.append("\nEVIDENCE WITH CONTEXTUAL DIFFERENCES:")
        for i, item in enumerate(evidence_summary.contextual[:max_evidence_per_group], 1):
            prompt_parts.append(_format_evidence_item(item.to_dict(), i))
            if item.context_analysis:
                for diff in item.context_analysis.differences[:3]:
                    prompt_parts.append(
                        f"    ↳ [{diff.dimension.upper()}] {diff.description}"
                    )

    # --- Instruction ---
    prompt_parts.append("\n" + "=" * 60)
    prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
    prompt_parts.append(
        "Based on the pre-analyzed evidence above, write a clear, "
        "evidence-grounded answer to the question. "
        "Structure your response as:\n"
        "1. DIRECT ANSWER — answer the question directly\n"
        "2. SUPPORTING EVIDENCE — what studies support this\n"
        "3. CONTRADICTING EVIDENCE — what studies contradict this (if any)\n"
        "4. CONTEXTUAL CONSIDERATIONS — important differences between studies\n"
        "5. OVERALL ASSESSMENT — summarize evidence quality and certainty\n\n"
        "Use [Author Year, Journal] citations for all key claims. "
        "Use hedged language ('evidence suggests', 'studies indicate') where appropriate. "
        "If evidence is INCONCLUSIVE, clearly state why."
    )
    prompt_parts.append("=" * 60)

    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Baseline RAG prompt (for comparison)
# ---------------------------------------------------------------------------

def build_baseline_prompt(question: str, retrieved_texts: List[str]) -> str:
    """
    Build a simple RAG prompt for the baseline comparison.

    This is the STANDARD RAG baseline: no NLI, no claim extraction,
    no evidence analysis — just retrieved passages + LLM.
    """
    prompt_parts = [
        "You are a medical research assistant. Answer the following question "
        "based on the retrieved biomedical literature passages. "
        "Cite the sources where possible.\n",
        f"QUESTION: {question}\n",
        "RETRIEVED PASSAGES:",
    ]
    for i, text in enumerate(retrieved_texts[:8], 1):
        prompt_parts.append(f"[Passage {i}]: {text[:600]}")

    prompt_parts.append(
        "\nAnswer the question based on these passages. "
        "Note any conflicts or uncertainties in the evidence."
    )
    return "\n".join(prompt_parts)
