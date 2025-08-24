#!/usr/bin/env python3
"""CLI for plotting Natural-Plan evaluation results."""
import argparse
from pathlib import Path
from ..analysis import ResultsAnalyzer
from .accuracy import plot_accuracy
from ..config import TASKS, MODELS, DEFAULT_ARGS


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Natural-Plan plots")
    p.add_argument("--results-dir", default=DEFAULT_ARGS["results_dir"], help="Results root directory")
    p.add_argument("--output-dir", default=DEFAULT_ARGS["output_dir"] + "plots", help="Plots output directory")
    p.add_argument("--task", choices=TASKS + ["all"], default="all",
                   help="Task to plot; 'all' generates combined plot too")
    p.add_argument("--models", default=DEFAULT_ARGS["models"], help="Comma-separated models to include")
    args = p.parse_args()

    analyzer = ResultsAnalyzer(Path(args.results_dir))
    results = analyzer.scan_all_results()
    if not results:
        print("No results found; nothing to plot")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models.split(",") if args.models else MODELS
    if args.task == "all":
        plot_accuracy(results, task=None, models=models, output_dir=output_dir)
        for task in TASKS:
            plot_accuracy(results, task=task, models=models, output_dir=output_dir)
    else:
        plot_accuracy(results, task=args.task, models=models, output_dir=output_dir)

    print(f"Plots written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


