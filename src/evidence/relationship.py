# =============================================================================
# src/evidence/relationship.py
#
# Evidence relationship aggregation.
#
# Takes the full set of NLI results and context analyses and produces
# a structured summary of the evidence relationships for the final answer.
#
# Evidence relationships per chunk:
#   - Supports: entailment with sufficient confidence
#   - Contradicts: contradiction with sufficient confidence
#   - Contextual difference: contradiction, but with identified contextual factors
#   - Neutral: neither supports nor contradicts
#
# This is distinct from the final Evidence Status (status.py) which gives
# the overall SUPPORTED / REFUTED / INCONCLUSIVE verdict.
# =============================================================================

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

from src.config import settings
from src.preprocessing.chunker import Chunk
from src.claims.claim_extractor import StructuredClaim
from src.nli.nli_classifier import NLIResult
from src.context.contradiction_analyzer import ContextComparisonResult
from src.evidence.grader import GRADEAssessment


# ---------------------------------------------------------------------------
# Evidence item — one chunk's full analysis
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """
    The complete analysis for one evidence chunk:
    source metadata + claim + NLI result + context analysis + GRADE certainty.
    """
    chunk: Chunk
    claim: StructuredClaim
    nli_result: NLIResult
    relationship: str                         # Supports | Contradicts | Contextual Difference | Neutral
    context_analysis: Optional[ContextComparisonResult] = None
    grade: Optional[GRADEAssessment] = None
    comparative_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "chunk_id": self.chunk.chunk_id,
            "paper_id": self.chunk.paper_id,
            "title": self.chunk.title,
            "authors": self.chunk.authors,
            "year": self.chunk.year,
            "journal": self.chunk.journal,
            "section": self.chunk.section,
            "study_type": self.chunk.study_type,
            "source": self.chunk.source,
            "text": self.chunk.text,
            "claim": self.claim.to_dict(),
            "nli": self.nli_result.to_dict(),
            "relationship": self.relationship,
            "context_analysis": self.context_analysis.to_dict() if self.context_analysis else None,
            "grade": self.grade.to_dict() if self.grade else None,
            "comparative_rationale": self.comparative_rationale,
        }
        return d


# ---------------------------------------------------------------------------
# Relationship derivation
# ---------------------------------------------------------------------------

def derive_relationship(
    nli_result: NLIResult,
    context_analysis: Optional[ContextComparisonResult] = None,
    claim: Optional[StructuredClaim] = None,
    query_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Derive a human-readable relationship label for one evidence item.

    Rules:
    1. ENTAILMENT + high confidence → "Supports"
       - If claim and query_context provided, verify intervention is not mismatched.
    2. CONTRADICTION + high confidence:
       - If context_analysis and context_analysis.conflict_type == "CONTEXTUAL" → "Contextual Difference"
       - Else → "Contradicts"
    3. NEUTRAL → "Neutral"
    4. Low-confidence ENTAILMENT/CONTRADICTION → downgrade to "Neutral"
    """
    label = nli_result.label
    conf = nli_result.confidence

    if label == "ENTAILMENT":
        if conf >= settings.nli_entailment_threshold:
            # Safeguard: if claim and query_context provided, check intervention mismatch
            if claim and query_context:
                target_ints = query_context.get("expanded_interventions") or query_context.get("interventions", [])
                if target_ints:
                    c_int = claim.intervention.lower()
                    c_text = claim.raw_text.lower()
                    if not any(ti in c_int or ti in c_text for ti in target_ints):
                        return "Neutral"
            return "Supports"
        else:
            return "Neutral"  # Low-confidence entailment treated as neutral

    elif label == "CONTRADICTION":
        if conf >= settings.nli_contradiction_threshold:
            if context_analysis and context_analysis.conflict_type == "CONTEXTUAL":
                return "Contextual Difference"
            return "Contradicts"
        else:
            return "Neutral"  # Low-confidence contradiction treated as neutral

    else:  # NEUTRAL
        return "Neutral"


# ---------------------------------------------------------------------------
# Build evidence item list
# ---------------------------------------------------------------------------

def build_evidence_items(
    chunks: List[Chunk],
    claims: List[StructuredClaim],
    nli_results: List[NLIResult],
    context_analyses: List[ContextComparisonResult],
    grades: Optional[List[GRADEAssessment]] = None,
    query_context: Optional[Dict[str, Any]] = None,
) -> List[EvidenceItem]:
    """
    Build a list of EvidenceItems by combining all analysis components.

    Args:
        chunks: Reranked evidence chunks.
        claims: Structured claims extracted from chunks (same order).
        nli_results: NLI results for each chunk (same order).
        context_analyses: Context comparisons for contradiction items.
                          Keyed by chunk_id.
        grades: GRADE assessments for each chunk (same order).
        query_context: Optional query normalization context, containing comparative_info.

    Returns:
        List of EvidenceItem objects in the same order as chunks.
    """
    # Check for comparative query context
    comp_info = None
    if query_context and query_context.get("comparative_info", {}).get("is_comparative"):
        from src.claims.comparative_gate import ComparativeQueryInfo, evaluate_comparative_chunk
        c_dict = query_context["comparative_info"]
        comp_info = ComparativeQueryInfo(
            is_comparative=True,
            comparative_type=c_dict.get("comparative_type", "NON_COMPARATIVE"),
            intervention_a=c_dict.get("intervention_a", ""),
            intervention_b=c_dict.get("intervention_b", ""),
            intervention_a_terms=c_dict.get("intervention_a_terms", []),
            intervention_b_terms=c_dict.get("intervention_b_terms", []),
            target_outcome=c_dict.get("target_outcome", ""),
            target_outcome_terms=c_dict.get("target_outcome_terms", []),
            population=c_dict.get("population", ""),
            claimed_superior=c_dict.get("claimed_superior"),
            raw_query=c_dict.get("raw_query", ""),
        )

    # Map context analyses by claim_b_id for quick lookup
    context_map: Dict[str, ContextComparisonResult] = {
        ca.claim_b_id: ca for ca in context_analyses
    }

    grade_list = grades if grades and len(grades) == len(chunks) else [None] * len(chunks)

    items: List[EvidenceItem] = []
    for chunk, claim, nli, gr in zip(chunks, claims, nli_results, grade_list):
        context_analysis = context_map.get(chunk.chunk_id)
        relationship = derive_relationship(nli, context_analysis, claim=claim, query_context=query_context)
        comp_rationale = None

        if comp_info and comp_info.is_comparative:
            relationship, comp_rationale = evaluate_comparative_chunk(
                chunk=chunk,
                claim=claim,
                comp_info=comp_info,
                nli_label=nli.label,
                nli_confidence=nli.confidence,
            )

        items.append(EvidenceItem(
            chunk=chunk,
            claim=claim,
            nli_result=nli,
            relationship=relationship,
            context_analysis=context_analysis,
            grade=gr,
            comparative_rationale=comp_rationale,
        ))

    return items


# ---------------------------------------------------------------------------
# Evidence relationship summary
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRelationshipSummary:
    """Summary of evidence relationships across all evidence items."""
    total: int
    supporting: List[EvidenceItem] = field(default_factory=list)
    contradicting: List[EvidenceItem] = field(default_factory=list)
    contextual: List[EvidenceItem] = field(default_factory=list)
    neutral: List[EvidenceItem] = field(default_factory=list)

    @property
    def n_supporting(self) -> int:
        return len(self.supporting)

    @property
    def n_contradicting(self) -> int:
        return len(self.contradicting)

    @property
    def n_contextual(self) -> int:
        return len(self.contextual)

    @property
    def n_neutral(self) -> int:
        return len(self.neutral)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "n_supporting": self.n_supporting,
            "n_contradicting": self.n_contradicting,
            "n_contextual": self.n_contextual,
            "n_neutral": self.n_neutral,
            "supporting": [i.to_dict() for i in self.supporting],
            "contradicting": [i.to_dict() for i in self.contradicting],
            "contextual": [i.to_dict() for i in self.contextual],
            "neutral": [i.to_dict() for i in self.neutral],
        }


def summarize_evidence_relationships(
    items: List[EvidenceItem],
) -> EvidenceRelationshipSummary:
    """
    Group evidence items by relationship type into a summary.
    """
    supporting = [i for i in items if i.relationship == "Supports"]
    contradicting = [i for i in items if i.relationship == "Contradicts"]
    contextual = [i for i in items if i.relationship == "Contextual Difference"]
    neutral = [i for i in items if i.relationship == "Neutral"]

    return EvidenceRelationshipSummary(
        total=len(items),
        supporting=supporting,
        contradicting=contradicting,
        contextual=contextual,
        neutral=neutral,
    )
