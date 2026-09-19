import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time
from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.generation.synthesizer import Synthesizer
from src.claims.claim_extractor import extract_claims, StructuredClaim
from src.claims.claim_grouper import ClaimGrouper
from src.claims.query_normalizer import normalize_query
from src.nli.nli_classifier import classify_evidence_against_query
from src.context.contradiction_analyzer import analyze_contradictions
from src.evidence.grader import GRADEGrader
from src.evidence.relationship import build_evidence_items, summarize_evidence_relationships
from src.evidence.status import determine_evidence_status, _distinct_study_ids
from src.preprocessing.chunker import Chunk

bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
embedder = Embedder()
vector_store = VectorStore()
bm25 = BM25Index.load(bm25_path)
synth = Synthesizer(embedder, vector_store, bm25, llm_client=None)

def print_trace_details(test_name, query, res):
    print("=" * 80)
    print(f"TRACE: {test_name}")
    print(f"Query: '{query}'")
    print("=" * 80)
    
    s_dict = res.status_result_dict
    ev_dict = res.evidence_summary_dict
    
    print(f"\n[FINAL VERDICT]: {res.status}")
    print(f"[CONFIDENCE]: {s_dict.get('confidence_label')} ({s_dict.get('confidence_score')})")
    print(f"[WEIGHTS]: W_Support={s_dict.get('weighted_support')} | W_Contradiction={s_dict.get('weighted_contradiction')} | W_Contextual={s_dict.get('contextual_weight')}")
    print(f"[EVIDENCE COUNTS]: Supporting={s_dict.get('n_supporting')} | Contradicting={s_dict.get('n_contradicting')} | Contextual={s_dict.get('n_contextual')} | Neutral={s_dict.get('n_neutral')}")
    print(f"[RATIONALE]: {s_dict.get('rationale')}")
    
    print("\n--- 1. TOP-10 RERANKED PASSAGES ---")
    for idx, r in enumerate(res.reranked):
        cid = r.get("chunk_id")
        pid = r.get("paper_id")
        score = r.get("reranker_score", 0.0)
        title = r.get("title") or "Unknown Title"
        text = r.get("text", "")[:180].replace("\n", " ")
        print(f"Passage #{idx+1}: [{cid}] (PMID/Paper: {pid}, Score: {score:.3f})")
        print(f"   Title: {title[:70]}")
        print(f"   Text: {text}...\n")
        
    print("--- 2. NLI OUTPUT PER PASSAGE ---")
    # Inspect each evidence item
    sup_items = ev_dict.get("supporting", [])
    con_items = ev_dict.get("contradicting", [])
    ctx_items = ev_dict.get("contextual", [])
    neu_items = ev_dict.get("neutral", [])
    
    all_items = []
    for it in sup_items:
        all_items.append((it, "Supporting"))
    for it in con_items:
        all_items.append((it, "Contradicting"))
    for it in ctx_items:
        all_items.append((it, "Contextual Difference"))
    for it in neu_items:
        all_items.append((it, "Neutral"))
        
    for it, rel in all_items:
        cid = it.get("chunk_id")
        nli = it.get("nli", {})
        label = nli.get("label")
        conf = nli.get("confidence", 0.0)
        print(f"Chunk [{cid}]: NLI={label} (conf: {conf:.3f}) -> Routed to: {rel}")
        
        ca = it.get("context_analysis")
        if ca:
            diffs = ca.get("differences", [])
            ctype = ca.get("conflict_type")
            print(f"   Context Analysis: conflict_type={ctype}, differences count={len(diffs)}")
            for d in diffs:
                print(f"     * [{d.get('dimension').upper()}]: {d.get('description')}")
        print()

print("\n" + "#" * 80)
print("# TEST 1: REVERSED METFORMIN QUERY (LIVE END-TO-END)")
print("#" * 80)
q1 = "Does metformin increase the risk of cardiovascular mortality in patients with Type 2 Diabetes?"
res1 = synth.run(q1, mode="diaveritias")
print_trace_details("TEST 1: REVERSED METFORMIN", q1, res1)

print("\n" + "#" * 80)
print("# TEST 2: FORWARD METFORMIN QUERY (LIVE END-TO-END)")
print("#" * 80)
q2 = "Does metformin reduce cardiovascular mortality in patients with Type 2 Diabetes?"
res2 = synth.run(q2, mode="diaveritias")
print_trace_details("TEST 2: FORWARD METFORMIN", q2, res2)

print("\n" + "#" * 80)
print("# TEST 3: SYNTHETIC REFUTED REACHABILITY (LIVE PIPELINE INFERENCE)")
print("#" * 80)
# Run synthetic case through the actual live pipeline models (real DeBERTa-v3, real extractor, real grouper, real analyzer, real status)
q3 = "Does metformin increase the risk of cardiovascular mortality in patients with Type 2 Diabetes?"
norm_q3 = normalize_query(q3)

chunk_syn1 = Chunk(
    chunk_id="synthetic_trial_01_results",
    paper_id="PMID_90000001",
    pmc_id="PMC90000001",
    title="Cardiovascular Mortality in Type 2 Diabetes: A Randomized Controlled Trial of Metformin",
    authors=["Smith J et al."],
    year="2023",
    journal="Lancet Diabetes & Endocrinology",
    doi="10.1016/S2213-8587(23)00001-X",
    section="Results",
    chunk_index=0,
    study_type="Randomized Controlled Trial",
    source="pmc_xml",
    text="In this randomized controlled trial of 4,500 patients with type 2 diabetes, metformin therapy significantly reduced the risk of cardiovascular mortality compared to control (HR 0.68, 95% CI 0.53-0.87, p=0.002).",
    token_count=130,
)

chunk_syn2 = Chunk(
    chunk_id="synthetic_trial_02_results",
    paper_id="PMID_90000002",
    pmc_id="PMC90000002",
    title="Long-term Cardiovascular Outcomes with Metformin Monotherapy in T2D Patients",
    authors=["Brown A et al."],
    year="2024",
    journal="New England Journal of Medicine",
    doi="10.1056/NEJMoa2300002",
    section="Results",
    chunk_index=0,
    study_type="Randomized Controlled Trial",
    source="pmc_xml",
    text="Metformin monotherapy resulted in a significant 28% reduction in cardiovascular death among adult patients with type 2 diabetes followed for five years.",
    token_count=120,
)

syn_chunks = [chunk_syn1, chunk_syn2]

# 1. Claim extraction
syn_claims = extract_claims(syn_chunks, llm_client=None)

# 2. Claim grouping & concept gate
grouper = ClaimGrouper()
on_target, neutral_gated, groups = grouper.group_and_filter_claims(syn_claims, norm_q3)

# 3. Live DeBERTa-v3 NLI inference
query_claim_text = norm_q3.get("hypothesis_text") or q3
nli_results = classify_evidence_against_query(query_claim_text, on_target)

# 4. Contextual contradiction analysis
ref_claim = StructuredClaim(
    chunk_id="query_ref",
    paper_id="query",
    section="Query",
    intervention="metformin",
    outcome="cardiovascular mortality",
    population="Type 2 Diabetes",
    direction="Increase",
    raw_text=q3,
)
context_analyses = analyze_contradictions(syn_claims, nli_results, reference_claim=ref_claim)

# 5. GRADE Grading
grader = GRADEGrader()
grades = [grader.grade_evidence(c, cl) for c, cl in zip(syn_chunks, syn_claims)]

# 6. Evidence relationships
ev_items = build_evidence_items(syn_chunks, syn_claims, nli_results, context_analyses, grades=grades)
ev_summary = summarize_evidence_relationships(ev_items)

# 7. Status determination
status_syn = determine_evidence_status(ev_summary, avg_nli_confidence=0.95, avg_retrieval_relevance=0.92)

print(f"[SYNTHETIC PIPELINE VERDICT]: {status_syn.status}")
print(f"[CONFIDENCE]: {status_syn.confidence_label} ({status_syn.confidence_score})")
print(f"[WEIGHTS]: W_Support={status_syn.weighted_support} | W_Contradiction={status_syn.weighted_contradiction} | W_Contextual={status_syn.contextual_weight}")
print(f"[EVIDENCE COUNTS]: Supporting={status_syn.n_supporting} | Contradicting={status_syn.n_contradicting} | Contextual={status_syn.n_contextual} | Neutral={status_syn.n_neutral}")
print(f"[RATIONALE]: {status_syn.rationale}")

for idx, (c, n, ca) in enumerate(zip(syn_chunks, nli_results, context_analyses), 1):
    print(f"\nSynthetic Study #{idx} [{c.paper_id}]:")
    print(f"   Text: {c.text}")
    print(f"   DeBERTa-v3 NLI: {n.label} (confidence: {n.confidence:.3f})")
    print(f"   Contradiction Analyzer: conflict_type={ca.conflict_type}, differences count={len(ca.differences)}")
