import argparse
import json
from pathlib import Path

import opik


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph RAG app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Index the source PDF into Qdrant Cloud")
    subparsers.add_parser("seed", help="Create or update the Opik evaluation dataset")
    subparsers.add_parser("evaluate", help="Run the Opik evaluation experiment")
    loop_parser = subparsers.add_parser("loop", help="Run configured Opik experiments")
    loop_parser.add_argument("--config", default="evals/configs.json")
    experiment_parser = subparsers.add_parser("experiment-loop", help="Run the experiment loop")
    experiment_parser.add_argument("--config", default="evals/experiments.json")
    run_parser = subparsers.add_parser(
        "run", help="Run one experiment, persist it, and decide keep/reject/revert"
    )
    run_parser.add_argument("name")
    run_parser.add_argument("--config", default="evals/experiments.json")
    report_parser = subparsers.add_parser("report", help="Show the worst-scoring cases for an experiment")
    report_parser.add_argument("name")
    report_parser.add_argument("--worst", type=int, default=5)
    report_parser.add_argument("--metric", default=None)
    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed PDF")
    ask_parser.add_argument("question")
    args = parser.parse_args()

    try:
        if args.command == "index":
            from .indexing import index_pdf

            print(f"Indexed {index_pdf()} chunks.")
        elif args.command == "seed":
            from .evals import create_seed_dataset

            print(f"Dataset version: {create_seed_dataset()}")
        elif args.command == "evaluate":
            from .evals import run_evaluation

            print(run_evaluation())
        elif args.command == "loop":
            from .evals import run_config_loop

            configurations = json.loads(Path(args.config).read_text())
            for result in run_config_loop(configurations):
                print(result)
        elif args.command == "experiment-loop":
            from .loop import run_experiment_loop

            for result in run_experiment_loop(Path(args.config)):
                print(result)
        elif args.command == "run":
            from .loop import run_iteration

            verdict = run_iteration(args.name, Path(args.config))
            print(f"{verdict.decision}: {verdict.reason}")
            for comparison in verdict.comparisons:
                print(f"  {comparison.metric}: {comparison.baseline} -> {comparison.current} (delta {comparison.delta})")
            if verdict.plateaued:
                print("Plateaued: no improvement over the last few kept runs. Consider a bigger change.")
        elif args.command == "report":
            from .report import load_and_format

            print(load_and_format(args.name, n=args.worst, metric=args.metric))
        else:
            from .rag import answer_question

            result = answer_question(args.question)
            print(result["answer"])
    finally:
        opik.flush_tracker()
