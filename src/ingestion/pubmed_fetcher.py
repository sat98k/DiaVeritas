# =============================================================================
# src/ingestion/pubmed_fetcher.py
#
# Downloads Type 2 Diabetes treatment literature from PubMed / PMC.
#
# Prefers structured XML (PMC full-text) when available.
# Falls back to PubMed abstract-only XML for papers not in PMC.
#
# Uses Biopython's Entrez interface, which respects NCBI rate limits:
#   - Without API key: 3 requests/second
#   - With NCBI_API_KEY in .env: 10 requests/second
#
# IMPORTANT: This module downloads metadata and XML/PDF files only.
# It does NOT parse or chunk. That is preprocessing's responsibility.
# =============================================================================

from __future__ import annotations

import time
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field

from loguru import logger

try:
    from Bio import Entrez, Medline
except ImportError:
    raise ImportError("biopython is required. Run: py -m pip install biopython")

import requests

from src.config import settings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PaperRecord:
    """
    Minimal metadata record for a paper.

    All fields that may be missing from the source use "Not reported"
    or "Unknown" — never fabricated values.
    """
    paper_id: str           # PubMed ID (PMID)
    pmc_id: Optional[str]   # PMC ID if available (e.g. "PMC1234567")
    title: str
    authors: List[str]
    year: str
    journal: str
    doi: Optional[str]
    abstract: str
    study_type: str         # "Not reported" if unknown
    source: str             # "pubmed_xml" | "pmc_xml" | "pdf_fallback"
    local_xml_path: Optional[str] = None   # path after download
    local_pdf_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Entrez setup
# ---------------------------------------------------------------------------

def _configure_entrez() -> None:
    """Configure Biopython Entrez with credentials from settings."""
    Entrez.email = settings.ncbi_email
    if settings.ncbi_api_key:
        Entrez.api_key = settings.ncbi_api_key
        logger.debug("NCBI API key set — rate limit: 10 req/s")
    else:
        logger.warning(
            "No NCBI_API_KEY set. Rate limit: 3 req/s. "
            "Add NCBI_API_KEY to .env for faster downloads."
        )


# ---------------------------------------------------------------------------
# PubMed search
# ---------------------------------------------------------------------------

# Default search query for T2D treatment literature.
# Designed to retrieve high-quality treatment studies.
DEFAULT_T2D_QUERY = (
    '(("type 2 diabetes"[Title/Abstract] OR "type II diabetes"[Title/Abstract] '
    'OR "T2DM"[Title/Abstract] OR "T2D"[Title/Abstract] OR "prediabetes"[Title/Abstract]) '
    'AND ("treatment"[Title/Abstract] OR "therapy"[Title/Abstract] '
    'OR "medication"[Title/Abstract] OR "pharmacotherapy"[Title/Abstract] '
    'OR "metformin"[Title/Abstract] OR "SGLT2"[Title/Abstract] '
    'OR "GLP-1"[Title/Abstract] OR "insulin"[Title/Abstract] '
    'OR "exercise"[Title/Abstract] OR "physical activity"[Title/Abstract] '
    'OR "lifestyle"[Title/Abstract] OR "diet"[Title/Abstract] '
    'OR "cardiovascular"[Title/Abstract] OR "HbA1c"[Title/Abstract]) '
    'AND ("clinical trial"[Publication Type] OR "randomized"[Title/Abstract] '
    'OR "systematic review"[Title/Abstract] OR "meta-analysis"[Title/Abstract]))'
)


# Landmark foundational clinical trials that define modern Type 2 Diabetes guidelines
LANDMARK_T2D_PMIDS = [
    "11832527",  # DPP 2002: Lifestyle Intervention vs Metformin in Prediabetes (NEJM)
    "11337921",  # Finnish DPS 2001: Prevention of T2D by lifestyle changes (NEJM)
    "19875686",  # DPPOS 2009: 10-year follow-up of diabetes incidence (Lancet)
    "26377189",  # DPPOS 2015: 15-year follow-up of lifestyle vs metformin (Lancet Diabetes)
    "21098771",  # HART-D 2010: Aerobic and resistance training on HbA1c in T2D (JAMA)
    "17876020",  # DARE 2007: Aerobic training, resistance training, or both in T2D (Ann Intern Med)
    "21540559",  # Umpierre 2011: Structured exercise training and HbA1c in T2D meta-analysis (JAMA)
    "26378442",  # EMPA-REG 2015: Empagliflozin, Cardiovascular Outcomes, and Mortality (NEJM)
    "31535829",  # DAPA-HF 2019: Dapagliflozin in Patients with Heart Failure (NEJM)
    "9742977",   # UKPDS 34 1998: Effect of intensive blood-glucose control with metformin (Lancet)
    "27295427",  # LEADER 2016: Liraglutide and Cardiovascular Outcomes in Type 2 Diabetes (NEJM)
    "30146932",  # DECLARE-TIMI 58 2019: Dapagliflozin and Cardiovascular Outcomes in T2D (NEJM)
    "31475794",  # CREDENCE 2019: Canagliflozin and Renal Outcomes in Type 2 Diabetes (NEJM)
    "34449189",  # SURPASS-2 2021: Tirzepatide versus Semaglutide in Patients with T2D (NEJM)
    "29677495",  # SUSTAIN-6 2016: Semaglutide and Cardiovascular Outcomes in Patients with T2D (NEJM)
    "19092145",  # Look AHEAD 2008: Cardiovascular effects of intensive lifestyle intervention in T2D
    "32865377",  # EMPEROR-Reduced 2020: Cardiovascular and Renal Outcomes with Empagliflozin in Heart Failure (NEJM)
    "34449188",  # STEP 1 2021: Once-Weekly Semaglutide in Adults with Overweight or Obesity (NEJM)
    # Landmark nutrition, diet & lifestyle epidemiology
    "24414842",  # Ding 2014: Caffeinated and decaffeinated coffee consumption and T2D risk meta-analysis (Diabetes Care)
    "29590460",  # Carlstrom 2018: Coffee consumption and reduced risk of developing T2D meta-analysis (Nutr Rev)
    "24510893",  # Jiang 2014: Coffee and caffeine intake and incidence of T2D meta-analysis (Eur J Clin Nutr)
    "29897866",  # Estruch 2018: Primary Prevention of CVD with Mediterranean Diet - PREDIMED (NEJM)
    "20693348",  # Malik 2010: Sugar-sweetened beverages and risk of metabolic syndrome and T2D (Diabetes Care)
    "29221645",  # Lean 2018: DiRECT trial - Primary care weight management for remission of T2D (Lancet)
]

# Clinical domain queries for multi-pillar balanced corpus building
DOMAIN_QUERIES: Dict[str, str] = {
    "nutrition_diet": (
        '(("type 2 diabetes"[Title/Abstract] OR "T2D"[Title/Abstract] OR "prediabetes"[Title/Abstract]) '
        'AND ("coffee"[Title/Abstract] OR "caffeine"[Title/Abstract] OR "diet"[Title/Abstract] '
        'OR "sugar"[Title/Abstract] OR "beverage"[Title/Abstract] OR "carbohydrate"[Title/Abstract] '
        'OR "mediterranean"[Title/Abstract] OR "fasting"[Title/Abstract] OR "fiber"[Title/Abstract]) '
        'AND ("meta-analysis"[Title/Abstract] OR "systematic review"[Title/Abstract] '
        'OR "clinical trial"[Publication Type] OR "cohort"[Title/Abstract] OR "prospective"[Title/Abstract]))'
    ),
    "pharmacotherapy": (
        '(("type 2 diabetes"[Title/Abstract] OR "T2D"[Title/Abstract]) '
        'AND ("metformin"[Title/Abstract] OR "GLP-1"[Title/Abstract] OR "semaglutide"[Title/Abstract] '
        'OR "tirzepatide"[Title/Abstract] OR "SGLT2"[Title/Abstract] OR "empagliflozin"[Title/Abstract] '
        'OR "dapagliflozin"[Title/Abstract] OR "insulin"[Title/Abstract] OR "sulfonylurea"[Title/Abstract]) '
        'AND ("clinical trial"[Publication Type] OR "randomized"[Title/Abstract] OR "meta-analysis"[Title/Abstract]))'
    ),
    "exercise_lifestyle": (
        '(("type 2 diabetes"[Title/Abstract] OR "T2D"[Title/Abstract] OR "prediabetes"[Title/Abstract]) '
        'AND ("exercise"[Title/Abstract] OR "physical activity"[Title/Abstract] OR "aerobic"[Title/Abstract] '
        'OR "resistance training"[Title/Abstract] OR "walking"[Title/Abstract] OR "HIIT"[Title/Abstract] '
        'OR "sedentary"[Title/Abstract]) '
        'AND ("trial"[Title/Abstract] OR "randomized"[Title/Abstract] OR "meta-analysis"[Title/Abstract]))'
    ),
    "complications_cvd": (
        '(("type 2 diabetes"[Title/Abstract] OR "T2D"[Title/Abstract]) '
        'AND ("cardiovascular"[Title/Abstract] OR "heart failure"[Title/Abstract] OR "MACE"[Title/Abstract] '
        'OR "mortality"[Title/Abstract] OR "chronic kidney disease"[Title/Abstract] '
        'OR "nephropathy"[Title/Abstract] OR "retinopathy"[Title/Abstract]) '
        'AND ("clinical trial"[Publication Type] OR "randomized"[Title/Abstract] OR "meta-analysis"[Title/Abstract]))'
    ),
    "prediabetes_remission": (
        '(("prediabetes"[Title/Abstract] OR "impaired glucose tolerance"[Title/Abstract] '
        'OR "diabetes remission"[Title/Abstract] OR "prevention of diabetes"[Title/Abstract]) '
        'AND ("lifestyle"[Title/Abstract] OR "metformin"[Title/Abstract] OR "weight loss"[Title/Abstract] '
        'OR "remission"[Title/Abstract]) '
        'AND ("trial"[Title/Abstract] OR "randomized"[Title/Abstract] OR "meta-analysis"[Title/Abstract] OR "cohort"[Title/Abstract]))'
    ),
}



def search_pubmed(
    query: str = DEFAULT_T2D_QUERY,
    max_results: int = 50,
    min_year: int = 2000,
    sort: str = "relevance",
) -> List[str]:
    """
    Search PubMed and return a list of PMIDs sorted by relevance.

    Args:
        query: Entrez search query string.
        max_results: Maximum number of PMIDs to return.
        min_year: Filter to papers published >= this year.
        sort: Sort order ("relevance" or "pub_date").

    Returns:
        List of PMID strings.
    """
    _configure_entrez()
    full_query = f"{query} AND {min_year}[PDAT]:3000[PDAT]"

    logger.info(f"Searching PubMed: max={max_results}, sort={sort}")
    logger.debug(f"Query: {full_query}")

    handle = Entrez.esearch(db="pubmed", term=full_query, retmax=max_results, sort=sort, usehistory="y")
    record = Entrez.read(handle)
    handle.close()

    pmids = record["IdList"]
    logger.info(f"Found {len(pmids)} PMIDs")
    return pmids


# ---------------------------------------------------------------------------
# Fetch PubMed metadata
# ---------------------------------------------------------------------------

def fetch_pubmed_metadata(pmids: List[str]) -> List[PaperRecord]:
    """
    Fetch metadata (title, authors, year, journal, abstract, DOI) for a list of PMIDs.

    Returns a list of PaperRecord objects.
    Handles missing fields gracefully — never fabricates data.
    """
    _configure_entrez()
    records: List[PaperRecord] = []

    batch_size = 20
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        logger.info(f"Fetching metadata for PMIDs {i+1}–{i+len(batch)} of {len(pmids)}")

        medline_records = []
        for attempt in range(3):
            try:
                handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="medline", retmode="text")
                medline_records = list(Medline.parse(handle))
                handle.close()
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    logger.warning(f"Batch fetch failed for PMIDs {batch[:3]}...: {exc}. Attempting item-by-item.")
                    for single_pmid in batch:
                        try:
                            h = Entrez.efetch(db="pubmed", id=single_pmid, rettype="medline", retmode="text")
                            medline_records.extend(list(Medline.parse(h)))
                            h.close()
                            time.sleep(0.35)
                        except Exception as item_err:
                            logger.debug(f"Skipping unretrievable PMID {single_pmid}: {item_err}")

        for rec in medline_records:
            pmid = rec.get("PMID", "Unknown")
            title = rec.get("TI", "Not reported")
            authors_raw = rec.get("AU", [])
            authors = authors_raw if authors_raw else ["Not reported"]
            year = _extract_year(rec.get("DP", "Unknown"))
            journal = rec.get("JT", rec.get("TA", "Not reported"))
            abstract = rec.get("AB", "Not reported")
            doi = _extract_doi(rec)
            pmc_id = _extract_pmc_id(rec)
            study_type = _infer_study_type(rec)

            records.append(
                PaperRecord(
                    paper_id=pmid,
                    pmc_id=pmc_id,
                    title=title,
                    authors=authors,
                    year=year,
                    journal=journal,
                    doi=doi,
                    abstract=abstract,
                    study_type=study_type,
                    source="pubmed_metadata",
                )
            )

        time.sleep(0.35)  # Respect rate limit even with API key

    logger.info(f"Fetched metadata for {len(records)} papers")
    return records


# ---------------------------------------------------------------------------
# PMC XML download
# ---------------------------------------------------------------------------

def download_pmc_xml(record: PaperRecord, output_dir: Path) -> bool:
    """
    Download the full-text XML for a paper from PMC if a PMC ID is available.

    Returns True if successful, False otherwise.
    The XML is saved to output_dir/<pmc_id>.xml
    """
    if not record.pmc_id:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{record.pmc_id}.xml"

    if out_path.exists():
        logger.debug(f"PMC XML already exists: {out_path.name}")
        record.local_xml_path = str(out_path)
        record.source = "pmc_xml"
        return True

    _configure_entrez()
    try:
        handle = Entrez.efetch(db="pmc", id=record.pmc_id, rettype="full", retmode="xml")
        xml_content = handle.read()
        handle.close()

        if isinstance(xml_content, str):
            xml_content = xml_content.encode("utf-8")

        out_path.write_bytes(xml_content)
        record.local_xml_path = str(out_path)
        record.source = "pmc_xml"
        logger.info(f"Downloaded PMC XML: {out_path.name}")
        time.sleep(0.35)
        return True

    except Exception as e:
        logger.warning(f"PMC XML download failed for {record.pmc_id}: {e}")
        return False


def download_pubmed_abstract_xml(record: PaperRecord, output_dir: Path) -> bool:
    """
    Fallback: download PubMed abstract-only XML for papers not in PMC.
    This gives structured metadata but not full text.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pubmed_{record.paper_id}.xml"

    if out_path.exists():
        record.local_xml_path = str(out_path)
        record.source = "pubmed_abstract_xml"
        return True

    _configure_entrez()
    try:
        handle = Entrez.efetch(db="pubmed", id=record.paper_id, rettype="abstract", retmode="xml")
        xml_content = handle.read()
        handle.close()

        if isinstance(xml_content, str):
            xml_content = xml_content.encode("utf-8")

        out_path.write_bytes(xml_content)
        record.local_xml_path = str(out_path)
        record.source = "pubmed_abstract_xml"
        logger.info(f"Downloaded PubMed abstract XML: {out_path.name}")
        time.sleep(0.35)
        return True

    except Exception as e:
        logger.warning(f"PubMed abstract XML download failed for {record.paper_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Batch corpus download
# ---------------------------------------------------------------------------

def build_domain_diversified_corpus(
    total_papers: int = 2000,
    raw_dir: Optional[Path] = None,
    min_year: int = 2000,
) -> List[PaperRecord]:
    """
    Download a balanced multi-pillar corpus partitioned across clinical domains:
    1. Nutrition & Dietary Epidemiology (caffeine, coffee, sugar, Mediterranean, fiber)
    2. Pharmacotherapy (Metformin, GLP-1, SGLT2i, tirzepatide, semaglutide, insulin)
    3. Exercise & Lifestyle (aerobic, resistance, walking, HIIT, sedentary)
    4. Complications & Cardiovascular/Renal (MACE, CKD, heart failure, retinopathy)
    5. Prediabetes & Remission (prevention, weight loss, remission)
    """
    raw_dir = raw_dir or settings.raw_dir
    raw_dir = Path(raw_dir)

    num_domains = len(DOMAIN_QUERIES)
    papers_per_domain = max(25, total_papers // num_domains)

    collected_pmids: List[str] = list(LANDMARK_T2D_PMIDS)

    for domain_name, query_str in DOMAIN_QUERIES.items():
        logger.info(f"Searching domain: {domain_name} (target={papers_per_domain})")
        try:
            domain_pmids = search_pubmed(
                query=query_str,
                max_results=papers_per_domain,
                min_year=min_year,
                sort="relevance",
            )
            for pmid in domain_pmids:
                if pmid not in collected_pmids:
                    collected_pmids.append(pmid)
        except Exception as e:
            logger.warning(f"Failed searching domain {domain_name}: {e}")

    final_pmids = collected_pmids[:total_papers]
    logger.info(
        f"Diversified corpus plan: {len(final_pmids)} total PMIDs "
        f"({len(LANDMARK_T2D_PMIDS)} landmarks + {len(final_pmids) - len(LANDMARK_T2D_PMIDS)} across {num_domains} clinical domains)"
    )

    records = fetch_pubmed_metadata(final_pmids)

    for rec in records:
        if rec.pmc_id:
            success = download_pmc_xml(rec, raw_dir)
            if success:
                continue
        download_pubmed_abstract_xml(rec, raw_dir)

    return records


def build_corpus(
    query: str = DEFAULT_T2D_QUERY,
    max_papers: int = 50,
    raw_dir: Optional[Path] = None,
    min_year: int = 2010,
) -> List[PaperRecord]:
    """
    Full corpus download pipeline.
    If max_papers >= 100 and default query is used, automatically balances
    across clinical domains (nutrition, pharmacotherapy, exercise, CVD, prediabetes).
    """
    if query == DEFAULT_T2D_QUERY and max_papers >= 100:
        return build_domain_diversified_corpus(
            total_papers=max_papers,
            raw_dir=raw_dir,
            min_year=min_year,
        )

    raw_dir = raw_dir or settings.raw_dir
    raw_dir = Path(raw_dir)

    pmids = search_pubmed(query=query, max_results=max_papers, min_year=min_year, sort="relevance")


    # Guarantee landmark foundational clinical trials are included
    combined_pmids = list(LANDMARK_T2D_PMIDS)
    for p in pmids:
        if p not in combined_pmids:
            combined_pmids.append(p)

    final_pmids = combined_pmids[:max_papers]
    logger.info(f"Final corpus plan: {len(final_pmids)} PMIDs ({len(LANDMARK_T2D_PMIDS)} landmarks + {len(final_pmids) - len(LANDMARK_T2D_PMIDS)} relevance-ranked)")

    records = fetch_pubmed_metadata(final_pmids)

    for rec in records:
        # Try PMC full-text first
        if rec.pmc_id:
            success = download_pmc_xml(rec, raw_dir)
            if success:
                continue
        # Fall back to PubMed abstract XML
        download_pubmed_abstract_xml(rec, raw_dir)

    return records


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_year(date_str: str) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", date_str)
    return match.group(0) if match else "Unknown"


def _extract_doi(rec: dict) -> Optional[str]:
    aids = rec.get("AID", [])
    for aid in aids:
        if "[doi]" in aid:
            return aid.replace("[doi]", "").strip()
    return None


def _extract_pmc_id(rec: dict) -> Optional[str]:
    pmc_ids = rec.get("PMC", [])
    if pmc_ids:
        raw = pmc_ids if isinstance(pmc_ids, str) else pmc_ids[0]
        if not raw.startswith("PMC"):
            raw = f"PMC{raw}"
        return raw
    # Also check in AID
    for aid in rec.get("AID", []):
        if "[pmc]" in aid.lower():
            return aid.split("[")[0].strip()
    return None


def _infer_study_type(rec: dict) -> str:
    """
    Infer study type from publication type tags.
    Returns a human-readable string or 'Not reported'.
    """
    pub_types = rec.get("PT", [])
    if not pub_types:
        return "Not reported"

    type_map = {
        "Randomized Controlled Trial": "RCT",
        "Clinical Trial": "Clinical Trial",
        "Meta-Analysis": "Meta-Analysis",
        "Systematic Review": "Systematic Review",
        "Review": "Review",
        "Observational Study": "Observational Study",
        "Cohort Study": "Cohort Study",
        "Case-Control Study": "Case-Control Study",
    }
    for pt in pub_types:
        for key, value in type_map.items():
            if key.lower() in pt.lower():
                return value

    return "Not reported"
