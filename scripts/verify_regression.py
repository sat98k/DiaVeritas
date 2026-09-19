import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.generation.synthesizer import Synthesizer

def main():
    bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
    embedder = Embedder()
    vector_store = VectorStore()
    bm25 = BM25Index.load(bm25_path)
    synth = Synthesizer(embedder, vector_store, bm25, llm_client=None)

    questions = [
        ("Q1", "Does metformin reduce HbA1c levels in patients with Type 2 Diabetes?"),
        ("Q2", "Do SGLT2 inhibitors reduce the risk of hospitalization for heart failure in patients with Type 2 Diabetes?"),
        ("Q2A", "What are the cardiovascular effects of SGLT2 inhibitors in patients with Type 2 Diabetes?"),
        ("Q2C", "Does empagliflozin reduce hospitalization for heart failure in Type 2 Diabetes?"),
        ("Q2D", "Does dapagliflozin reduce hospitalization for heart failure in Type 2 Diabetes?"),
        ("M1_Coffee_Sugar", "Does avoiding added sugar in coffee prevent progression from prediabetes to Type 2 Diabetes?"),
        ("M2_Rosiglitazone", "Does rosiglitazone increase the risk of myocardial infarction in patients with Type 2 Diabetes?"),
        ("M3_Reversed_Metformin", "Does metformin increase the risk of cardiovascular mortality in patients with Type 2 Diabetes?"),
    ]

    print("=" * 80)
    print("REGRESSION VERIFICATION: Q1, Q2, Q2A, Q2C, Q2D")
    print("=" * 80)

    for q_id, q_text in questions:
        print(f"\n=== Running {q_id}: '{q_text}' ===")
        res = synth.run(q_text, mode="diaveritias")
        s_dict = res.status_result_dict
        
        status = res.status
        conf_score = s_dict.get("confidence_score")
        conf_label = s_dict.get("confidence_label")
        sup_cnt = len(res.evidence_summary_dict.get("supporting", []))
        con_cnt = len(res.evidence_summary_dict.get("contradicting", []))
        ctx_cnt = len(res.evidence_summary_dict.get("contextual", []))
        neu_cnt = len(res.evidence_summary_dict.get("neutral", []))
        
        w_sup = s_dict.get("weighted_support")
        w_con = s_dict.get("weighted_contradiction")
        w_ctx = s_dict.get("contextual_weight")
        conf_breakdown = s_dict.get("confidence_breakdown", {})
        study_cap = conf_breakdown.get("study_count_cap", 1.0)
        
        # Grounding check
        ungrounded = getattr(res, "ungrounded_claims", [])
        stripped = getattr(res, "stripped_claims", [])

        print(f"Verdict: {status}")
        print(f"Confidence: {conf_label} ({conf_score}) [Study factor: {study_cap}]")
        print(f"Evidence counts: Supporting={sup_cnt}, Contradicting={con_cnt}, Contextual={ctx_cnt}, Neutral={neu_cnt}")
        print(f"Weights: W_Support={w_sup}, W_Contradiction={w_con}, W_Contextual={w_ctx}")
        print(f"Grounding: Ungrounded in answer={len(ungrounded)}, Stripped={len(stripped)}")
        print(f"Answer Preview:\n{res.answer[:250]}...")

if __name__ == "__main__":
    main()
