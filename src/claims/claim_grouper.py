# =============================================================================
# src/claims/claim_grouper.py
#
# Module 12 — Claim Grouping & Concept Gate
#
# Implements FR-12.1 through FR-12.4:
#   - Groups extracted claims by matching (Intervention, Population, Outcome)
#   - Prevents pairwise all-to-all NLI explosion (FR-12.1)
#   - Enforces the Concept Gate: claims with mismatched outcomes or unrelated
#     interventions are deterministically flagged as NEUTRAL with respect
#     to the query (FR-12.3) rather than being compared for contradiction.
#   - Deterministic and inspectable grouping (FR-12.4).
# =============================================================================

from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from loguru import logger

from src.claims.claim_extractor import StructuredClaim
from src.preprocessing.chunker import Chunk


@dataclass
class ClaimGroup:
    """
    A cluster of claims sharing the same clinical context (Intervention + Outcome).
    """
    group_id: str
    intervention: str
    outcome: str
    is_query_target: bool
    claims: List[StructuredClaim] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "intervention": self.intervention,
            "outcome": self.outcome,
            "is_query_target": self.is_query_target,
            "claim_count": len(self.claims),
            "rejection_reason": self.rejection_reason,
            "chunk_ids": [c.chunk_id for c in self.claims],
        }


class ClaimGrouper:
    """
    Groups extracted claims and filters out off-target claims before NLI.
    """

    def __init__(self):
        pass

    def group_and_filter_claims(
        self,
        claims: List[StructuredClaim],
        normalized_query: Dict[str, Any],
    ) -> Tuple[List[StructuredClaim], List[Tuple[StructuredClaim, str]], List[ClaimGroup]]:
        """
        Group claims by Intervention + Outcome, and filter for query compatibility.

        Args:
            claims: Extracted structured claims from reranked passages.
            normalized_query: Output from query_normalizer.normalize_query().

        Returns:
            Tuple of:
              - on_target_claims: claims eligible for NLI verification against query
              - neutral_claims: claims marked NEUTRAL with explicit reason
              - all_groups: inspectable list of ClaimGroup objects
        """
        target_interventions = [i.lower() for i in normalized_query.get("interventions", [])]
        target_outcomes = [o.lower() for o in normalized_query.get("outcomes", [])]

        groups: Dict[str, ClaimGroup] = {}
        on_target_claims: List[StructuredClaim] = []
        neutral_claims: List[Tuple[StructuredClaim, str]] = []

        for claim in claims:
            c_text_lower = claim.raw_text.lower()
            c_int = claim.intervention.lower() if claim.intervention != "Not reported" else ""
            c_out = claim.outcome.lower() if claim.outcome != "Not reported" else ""

            # Check intervention match: if target interventions exist, claim text/int must match at least one
            int_matched = True
            if target_interventions:
                int_matched = any(ti in c_text_lower or ti in c_int for ti in target_interventions)

            # Check outcome match: if target outcomes exist, claim text/out must match at least one
            out_matched = True
            if target_outcomes:
                out_matched = any(
                    to in c_text_lower or to in c_out or self._outcome_matches(to, c_text_lower)
                    for to in target_outcomes
                )

            # Determine canonical group key
            key_int = claim.intervention if claim.intervention != "Not reported" else "General T2D Care"
            key_out = claim.outcome if claim.outcome != "Not reported" else "Unspecified Endpoint"
            group_key = f"{key_int}::{key_out}"

            is_target = int_matched and out_matched

            if group_key not in groups:
                groups[group_key] = ClaimGroup(
                    group_id=group_key,
                    intervention=key_int,
                    outcome=key_out,
                    is_query_target=is_target,
                )
            groups[group_key].claims.append(claim)

            if is_target:
                on_target_claims.append(claim)
            else:
                reason = []
                if not int_matched and target_interventions:
                    reason.append(f"Intervention does not target '{', '.join(target_interventions)}'")
                if not out_matched and target_outcomes:
                    reason.append(f"Outcome differs from target '{', '.join(target_outcomes)}'")
                neutral_reason = "; ".join(reason) if reason else "Off-target clinical context"
                groups[group_key].rejection_reason = neutral_reason
                neutral_claims.append((claim, neutral_reason))

        logger.info(
            f"Claim grouping complete: {len(groups)} groups identified. "
            f"On-target: {len(on_target_claims)}, Neutral (off-target): {len(neutral_claims)}"
        )
        return on_target_claims, neutral_claims, list(groups.values())

    def _outcome_matches(self, target_outcome: str, text: str) -> bool:
        """Helper to match clinical synonyms for common target outcomes."""
        synonyms = {
            "cardiovascular events/risk": ["cardiovascular", "mace", "myocardial", "heart failure", "cv death", "stroke"],
            "glycemic control (hba1c/glucose)": ["hba1c", "a1c", "blood glucose", "glycemic", "fasting glucose"],
            "renal outcomes": ["kidney", "renal", "egfr", "albuminuria", "esrd", "nephropathy"],
            "mortality": ["mortality", "death", "survival", "fatal"],
            "weight/bmi": ["weight", "bmi", "body mass", "adiposity"],
        }
        for category, syn_list in synonyms.items():
            if target_outcome in category:
                if any(syn in text for syn in syn_list):
                    return True
        return False
