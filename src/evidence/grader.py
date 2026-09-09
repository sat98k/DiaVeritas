# =============================================================================
# src/evidence/grader.py
#
# Module 15 — Evidence Strength & Reliability Assessment (GRADE / ADA)
#
# Implements FR-15.1 through FR-15.4:
#   - Assigns each study/claim an evidence-certainty tier using GRADE / ADA
#     principles (High, Moderate, Low, Very Low).
#   - Evaluates study design hierarchy:
#       Systematic Review / Meta-Analysis > RCT > Cohort > Case-Control > Case Series / Review
#   - Evaluates sample size and risk of bias factors.
#   - Provides certainty weights for evidence aggregation (FR-15.4) so that
#     low-certainty studies cannot outvote landmark high-certainty trials.
# =============================================================================

from __future__ import annotations

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from loguru import logger

from src.preprocessing.chunker import Chunk
from src.claims.claim_extractor import StructuredClaim


# GRADE Certainty Tiers
GRADE_HIGH = "HIGH"          # Systematic reviews, meta-analyses, large landmark RCTs
GRADE_MODERATE = "MODERATE"  # Standard randomized controlled trials
GRADE_LOW = "LOW"            # Prospective cohort, observational studies, case-control
GRADE_VERY_LOW = "VERY_LOW"  # Case series, narrative reviews, editorials, animal/in vitro

# Methodological weights per tier
GRADE_WEIGHTS: Dict[str, float] = {
    GRADE_HIGH: 4.0,
    GRADE_MODERATE: 3.0,
    GRADE_LOW: 1.5,
    GRADE_VERY_LOW: 0.5,
}


@dataclass
class GRADEAssessment:
    """
    GRADE / ADA evidence certainty assessment for a single study/chunk.
    """
    chunk_id: str
    paper_id: str
    tier: str                    # HIGH | MODERATE | LOW | VERY_LOW
    weight: float                # Numerical aggregation weight (0.5 to 4.0)
    study_design: str            # Detected or declared study design
    risk_of_bias: str            # "Low" | "Moderate" | "High" | "Unclear"
    sample_size_note: str        # Notes on sample size or power
    rationale: str               # Explicit rationale explaining the tier assignment

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GRADEGrader:
    """
    Automated evidence-certainty grader implementing GRADE and ADA Standards of Care.
    """

    def grade_evidence(
        self,
        chunk: Chunk,
        claim: Optional[StructuredClaim] = None,
    ) -> GRADEAssessment:
        """
        Grade a chunk based on study type, context, and methodology indicators.
        """
        text_lower = chunk.text.lower()
        title_lower = chunk.title.lower()
        study_type = (chunk.study_type or "").strip()
        combined_text = f"{title_lower} {study_type.lower()} {text_lower[:600]}"

        # 1. Base tier from study design hierarchy
        if any(kw in combined_text for kw in ["systematic review", "meta-analysis", "meta analysis", "cochrane"]):
            tier = GRADE_HIGH
            design = "Systematic Review / Meta-Analysis"
            base_rationale = "Highest baseline certainty: systematic synthesis of multiple clinical trials."
            bias = "Low"
        elif any(kw in combined_text for kw in ["randomized", "randomised", "rct", "double-blind", "placebo-controlled", "clinical trial"]):
            tier = GRADE_MODERATE
            design = "Randomized Controlled Trial (RCT)"
            base_rationale = "High initial baseline certainty: randomized interventional design."
            bias = "Low"
        elif any(kw in combined_text for kw in ["cohort", "prospective", "observational", "longitudinal", "registry"]):
            tier = GRADE_LOW
            design = "Observational / Cohort Study"
            base_rationale = "Moderate-to-low baseline certainty: observational data subject to residual confounding."
            bias = "Moderate"
        elif any(kw in combined_text for kw in ["case-control", "case control", "cross-sectional", "retrospective"]):
            tier = GRADE_LOW
            design = "Retrospective / Case-Control Study"
            base_rationale = "Low baseline certainty: retrospective design prone to recall and selection bias."
            bias = "Moderate"
        else:
            tier = GRADE_VERY_LOW
            design = "Narrative Review / Expert Opinion / In Vitro"
            base_rationale = "Very low certainty: non-randomized or non-systematic clinical discussion."
            bias = "High"

        # 2. Sample size detection & tier adjustment
        sample_size_note = "Sample size not explicitly parsed"
        n_match = re.search(r"\b(?:n\s*=\s*|sample size of\s*|enrolled\s*)(\d{1,3}(?:,\d{3})+|\d{2,6})\b", text_lower)
        if n_match:
            try:
                n_val = int(n_match.group(1).replace(",", ""))
                sample_size_note = f"Sample size n={n_val:,}"
                if n_val >= 2000 and tier == GRADE_MODERATE:
                    tier = GRADE_HIGH
                    base_rationale += f" Upgraded to HIGH certainty due to large statistical power ({sample_size_note})."
                elif n_val < 60 and tier in (GRADE_MODERATE, GRADE_LOW):
                    tier = GRADE_VERY_LOW
                    bias = "High"
                    base_rationale += f" Downgraded to VERY LOW certainty due to small sample size ({sample_size_note})."
            except ValueError:
                pass

        # 3. Landmark trial recognition (automatic HIGH tier)
        landmark_titles = ["dpp", "ukpds", "empa-reg", "dapa-hf", "declare", "credence", "surpass", "sustain", "leader", "direct"]
        if any(lt in combined_text for lt in landmark_titles) and tier == GRADE_MODERATE:
            tier = GRADE_HIGH
            base_rationale += " Upgraded to HIGH certainty: landmark multi-center clinical outcome trial."

        # 4. Explicit Risk of Bias keyword check
        if any(hb in text_lower for hb in ["high risk of bias", "lack of blinding", "uncontrolled study", "significant attrition"]):
            bias = "High"
            if tier == GRADE_HIGH:
                tier = GRADE_MODERATE
            elif tier == GRADE_MODERATE:
                tier = GRADE_LOW

        weight = GRADE_WEIGHTS.get(tier, 1.0)

        return GRADEAssessment(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            tier=tier,
            weight=weight,
            study_design=design,
            risk_of_bias=bias,
            sample_size_note=sample_size_note,
            rationale=base_rationale,
        )


# Global singleton
grade_grader = GRADEGrader()
