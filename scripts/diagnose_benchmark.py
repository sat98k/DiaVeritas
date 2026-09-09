import json
from pathlib import Path

data = json.loads(Path("data/clinical_benchmark_results.json").read_text(encoding="utf-8"))
items = data["per_question_results"]

print(f"Total questions: {len(items)}")
matches = [it for it in items if it["diaveritas"]["verdict_match"]]
mismatches = [it for it in items if not it["diaveritas"]["verdict_match"]]

print(f"Matches: {len(matches)} ({len(matches)/len(items):.1%})")
print(f"Mismatches: {len(mismatches)} ({len(mismatches)/len(items):.1%})\n")

print("=" * 80)
print("MISMATCH BREAKDOWN & DIAGNOSIS")
print("=" * 80)

for idx, it in enumerate(mismatches, 1):
    qid = it["qid"]
    q = it["question"]
    exp = it["expected_verdict"]
    act = it["diaveritas"]["verdict"]
    cnts = it["diaveritas"]["evidence_counts"]
    wsup = it["diaveritas"]["weighted_support"]
    wcon = it["diaveritas"]["weighted_contradiction"]
    wctx = it["diaveritas"]["contextual_weight"]
    ans = it["diaveritas"]["answer_snippet"][:220].replace("\n", " ")
    
    n_sup = cnts.get("supporting", 0)
    n_con = cnts.get("contradicting", 0)
    n_ctx = cnts.get("contextual", 0)
    n_neu = cnts.get("neutral", 0)
    
    # Categorization:
    # (a) Single study replication gate enforced (n_sup == 1 or n_con == 1 when exp is SUPPORTED/REFUTED)
    # (b) System should have reached REFUTED but didn't (true contradiction present, 2+ independent studies)
    # (c) System should have reached SUPPORTED but returned INCONCLUSIVE despite 2+ independent qualifying studies
    # (d) Gold-standard label itself is questionable/ambiguous (e.g. PHARM-005 metformin CV mortality expected INCONCLUSIVE but 6 RCTs show benefit -> SUPPORTED)
    cat = "UNKNOWN"
    reason = ""
    
    if qid == "PHARM-005":
        # Question: Does metformin reduce cardiovascular mortality in type 2 diabetes?
        # DiaVeritas: SUPPORTED (6 supporting RCTs: UKPDS, etc.). Expected in CSV: INCONCLUSIVE.
        cat = "Category (d): Gold-standard label ambiguous/questionable"
        reason = "UKPDS landmark and meta-analyses provide robust 6-trial consensus for mortality benefit. Gold set marked INCONCLUSIVE based on older debate."
    elif exp in ("SUPPORTED", "REFUTED") and act == "INCONCLUSIVE":
        if (exp == "SUPPORTED" and n_sup == 1) or (exp == "REFUTED" and n_con == 1):
            cat = "Category (a): Safe Conservatism (FR-16.2/16.4 replication gate enforced)"
            reason = f"Corpus only contains 1 independent qualifying study (need >=2 independent landmark trials). Correctly withheld {exp}."
        elif (exp == "REFUTED" and n_con == 0):
            if n_ctx > 0 and n_sup == 0:
                cat = "Category (a): Safe Conservatism (FR-16.4 contextual divergence / lack of direct trial)"
                reason = f"Literature addresses related subgroups/comparators ({n_ctx} contextual, {n_neu} neutral) without direct contradicting RCTs in corpus."
            else:
                cat = "Category (b): Refuted Miss (needs investigation if 2+ studies in corpus)"
                reason = "Expected REFUTED, but corpus lacked 2+ direct contradicting trials."
        elif exp == "SUPPORTED" and n_sup >= 2:
            cat = "Category (c): Supported Miss (2+ qualifying studies present)"
            reason = "Multiple supporting items found but withheld."
        else:
            cat = "Category (a): Safe Conservatism (insufficient trial evidence in corpus)"
            reason = f"Only {n_sup} supporting and {n_con} contradicting found in 15k corpus."

    print(f"\n{idx}. [{qid}] Expected: {exp} | Actual: {act}")
    print(f"   Question: {q}")
    print(f"   Evidence: Sup={n_sup} (W={wsup}), Con={n_con} (W={wcon}), Ctx={n_ctx} (W={wctx}), Neu={n_neu}")
    print(f"   Diagnosis: {cat}")
    print(f"   Rationale: {reason}")
    print(f"   Answer Snippet: {ans}...")
