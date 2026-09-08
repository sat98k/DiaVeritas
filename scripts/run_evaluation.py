# =============================================================================
# scripts/run_evaluation.py
#
# Runs the DiaVeritas vs. baseline evaluation suite.
#
# Usage:
#   py scripts/run_evaluation.py
#   py scripts/run_evaluation.py --output results/eval_results.json
#   py scripts/run_evaluation.py --questions-file my_questions.txt
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import click
from loguru import logger

from src.config import settings


@click.command()
@click.option("--output", default=None, help="Output JSON file path for results.")
@click.option("--questions-file", default=None, help="Text file with one question per line.")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def main(output, questions_file, log_level):
    """Run DiaVeritas vs. baseline evaluation."""
    import sys as _sys
    logger.remove()
    logger.add(_sys.stderr, level=log_level)

    # Load pipeline
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_index import BM25Index
    from src.generation.synthesizer import Synthesizer
    from src.evaluation.baseline_runner import BaselineRunner, DEFAULT_TEST_QUESTIONS

    try:
        from src.generation.llm_client import LLMClient
        llm = LLMClient()
        llm._load()
    except Exception as e:
        logger.warning(f"LLM not available: {e}")
        llm = None

    bm25_path = Path(settings.processed_dir) / "bm25_index.pkl"
    embedder = Embedder()
    vector_store = VectorStore()
    bm25 = BM25Index.load(bm25_path)

    synthesizer = Synthesizer(embedder, vector_store, bm25, llm)
    runner = BaselineRunner(synthesizer)

    # Questions
    questions = DEFAULT_TEST_QUESTIONS
    if questions_file:
        with open(questions_file) as f:
            questions = [l.strip() for l in f if l.strip()]
        logger.info(f"Loaded {len(questions)} questions from {questions_file}")

    # Output path
    out_path = Path(output) if output else Path(settings.data_dir) / "evaluation_results.json"

    # Run
    runner.run_evaluation(questions=questions, output_path=out_path)


if __name__ == "__main__":
    main()
