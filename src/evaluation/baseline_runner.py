# =============================================================================
# src/evaluation/baseline_runner.py
#
# Runs evaluation comparing DiaVeritas vs. standard RAG baseline.
#
# For each test question:
#   1. Run DiaVeritas (full pipeline)
#   2. Run Baseline RAG (retrieval + direct LLM)
#   3. Record metrics for comparison
#
# Reports:
#   - Citation presence (DiaVeritas vs. Baseline)
#   - Evidence coverage (DiaVeritas vs. Baseline)
#   - Answer uncertainty language usage
#   - Evidence status consistency
#   - Latency comparison
#   - Side-by-side answers
#
# Test questions are a small curated set of T2D treatment questions
# designed to exercise the contradiction detection pipeline.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

from src.evaluation.metrics import evaluate_result


# ---------------------------------------------------------------------------
# Default test questions
# ---------------------------------------------------------------------------

DEFAULT_TEST_QUESTIONS = [
    "Does metformin reduce cardiovascular risk in patients with type 2 diabetes?",
    "Is SGLT2 inhibitor treatment associated with reduced heart failure hospitalization in T2D?",
    "Do GLP-1 receptor agonists improve HbA1c in type 2 diabetes patients?",
    "What is the effect of pioglitazone on cardiovascular outcomes in T2D?",
    "Does intensive glycemic control reduce microvascular complications in type 2 diabetes?",
    "Are SGLT2 inhibitors safe for patients with chronic kidney disease and T2D?",
    "Does semaglutide reduce major adverse cardiovascular events in T2D patients?",
    "What are the effects of bariatric surgery on glycemic control in T2D?",
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class BaselineRunner:
    """
    Runs DiaVeritas and baseline RAG side by side for evaluation.
    """

    def __init__(self, synthesizer):
        """
        Args:
            synthesizer: An initialized Synthesizer instance.
        """
        self._synthesizer = synthesizer

    def evaluate_question(self, question: str) -> Dict[str, Any]:
        """
        Run both pipeline modes for a single question.

        Returns:
            Dict with results and metrics for both modes.
        """
        logger.info(f"Evaluating: '{question}'")

        # DiaVeritas full pipeline
        diaveritias_result = self._synthesizer.run(question, mode="diaveritias")
        diaveritias_dict = diaveritias_result.to_dict()
        diaveritias_metrics = evaluate_result(
            diaveritias_dict,
            query_normalized=diaveritias_result.normalized_query,
        )

        # Baseline RAG
        baseline_result = self._synthesizer.run(question, mode="baseline")
        baseline_dict = baseline_result.to_dict()
        baseline_metrics = evaluate_result(
            baseline_dict,
            query_normalized=baseline_result.normalized_query,
        )

        return {
            "question": question,
            "diaveritias": {
                "answer": diaveritias_result.answer[:800],
                "status": diaveritias_result.status,
                "metrics": diaveritias_metrics,
                "latency": diaveritias_result.latency_seconds,
            },
            "baseline": {
                "answer": baseline_result.answer[:800],
                "metrics": baseline_metrics,
                "latency": baseline_result.latency_seconds,
            },
        }

    def run_evaluation(
        self,
        questions: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all questions and optionally save results to JSON.

        Args:
            questions: List of test questions. Defaults to DEFAULT_TEST_QUESTIONS.
            output_path: If provided, save results to this JSON file.

        Returns:
            List of evaluation result dicts.
        """
        questions = questions or DEFAULT_TEST_QUESTIONS
        results = []

        logger.info(f"Running evaluation on {len(questions)} questions...")
        for i, q in enumerate(questions, 1):
            logger.info(f"Question {i}/{len(questions)}")
            try:
                result = self.evaluate_question(q)
                results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed for question {i}: {e}")
                results.append({
                    "question": q,
                    "error": str(e),
                })

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to {output_path}")

        # Print summary
        _print_summary(results)

        return results


def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a brief summary of evaluation results."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results to summarize.")
        return

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for result in valid:
        q = result["question"][:60]
        dv = result["diaveritias"]
        bl = result["baseline"]

        print(f"\nQ: {q}...")
        print(f"  DiaVeritas | Status: {dv['status']} | "
              f"Citations: {dv['metrics']['citation_presence']:.2f} | "
              f"Latency: {dv['latency']:.1f}s")
        print(f"  Baseline   | Citations: {bl['metrics']['citation_presence']:.2f} | "
              f"Latency: {bl['latency']:.1f}s")

    # Aggregate
    dv_citations = [r["diaveritias"]["metrics"]["citation_presence"] for r in valid]
    bl_citations = [r["baseline"]["metrics"]["citation_presence"] for r in valid]
    dv_uncertainty = [r["diaveritias"]["metrics"]["has_uncertainty_language"] for r in valid]

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    print("\n" + "-" * 60)
    print("AGGREGATE METRICS")
    print(f"  DiaVeritas avg citation presence: {avg(dv_citations):.3f}")
    print(f"  Baseline    avg citation presence: {avg(bl_citations):.3f}")
    print(f"  DiaVeritas uncertainty language: {sum(dv_uncertainty)}/{len(dv_uncertainty)}")
    print("=" * 60)
