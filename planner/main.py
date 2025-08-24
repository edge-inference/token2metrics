#!/usr/bin/env python3
"""
Main analysis script for comprehensive Natural-Plan evaluation analysis.

This script runs all analysis types and generates plots for all tasks and models.

Usage:
    python -m token2metrics.planner.main
    python -m token2metrics.planner.main --results-dir custom_results/
    python -m token2metrics.planner.main --skip-plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .analysis import ResultsAnalyzer
from .processing.coverage import ScanResults
from .compare import scan_budget_results
from .plots.accuracy import plot_accuracy
from .config import TASKS, MODELS, DEFAULT_ARGS, PATHS, ensure_output_dirs


def main():
    parser = argparse.ArgumentParser(description="Natural-Plan analysis main entry point")
    parser.add_argument("--results-dir", default=DEFAULT_ARGS["results_dir"],
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default=DEFAULT_ARGS["output_dir"],
                       help="Directory to save all outputs")
    parser.add_argument("--mode", choices=["full", "plots", "compare", "summary"], default="full",
                       help="Analysis mode: full=comprehensive, plots=plotting only, compare=budget comparison, summary=summary only")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Skip generating plots (full mode only)")
    parser.add_argument("--skip-missing-check", action="store_true",
                       help="Skip checking for missing evaluations (full mode only)")
    parser.add_argument("--task", help="Filter by specific task (meeting, calendar, trip)")
    parser.add_argument("--models", help="Comma-separated list of models (14b,8b,1.5b)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        return
    
    # Parse models list
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    
    print(f"* Starting Natural-Plan analysis ({args.mode} mode)...")
    print("=" * 60)
    
    # Handle different modes
    if args.mode == "compare":
        _run_budget_comparison(results_dir, output_dir)
        return
    elif args.mode == "plots":
        _run_plotting_only(results_dir, output_dir, args.task, models)
        return
    elif args.mode == "summary":
        _run_summary_only(results_dir, output_dir)
        return
    
    # Create output directory structure
    dirs = ensure_output_dirs(output_dir)
    plots_dir = dirs["plots_dir"]
    
    # 1. Check for missing evaluations
    if not args.skip_missing_check:
        print("\n- Step 1: Checking for missing evaluations...")
        checker = ScanResults(results_dir)
        missing_baseline = checker.find_missing_baseline_evals()
        
        if missing_baseline:
            print(f"! Found {len(missing_baseline)} missing baseline evaluations:")
            for task, model in missing_baseline:
                print(f"    {task} task with {model} model")
            print("    Run baseline evaluations before proceeding with analysis.")
        else:
            print("✓ All baseline evaluations found!")
    
    print("\n- Step 2: Scanning all evaluation results...")
    analyzer = ResultsAnalyzer(results_dir)
    results = analyzer.scan_all_results()
    
    if not results:
        print("✗ No results found to analyze!")
        return
    
    print(f"Found {len(results)} evaluation results")
    
    print("\n- Step 3: Generating summary report...")
    analyzer.generate_summary_report(output_dir)
    
    if not args.skip_plots:
        print("\n- Step 4: Generating accuracy vs tokens plots...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Generate plots for each task
        tasks = TASKS
        models = MODELS
        
        for task in tasks:
            print(f"  - Creating plot for {task} task...")
            try:
                plot_accuracy(
                    results=results,
                    task=task,
                    models=models,
                    output_dir=plots_dir
                )
            except Exception as e:
                print(f"    ! Error creating plot for {task}: {e}")
        
        # Generate combined plot (all tasks)
        print("  - Creating combined plot for all tasks...")
        try:
            plot_accuracy(
                results=results,
                task=None,  # All tasks
                models=models,
                output_dir=plots_dir
            )
        except Exception as e:
            print(f"    ! Error creating combined plot: {e}")
    
    # 5. Generate analysis insights
    print("\n- Step 5: Generating analysis insights...")
    insights_file = output_dir / "analysis_insights.md"
    
    # Analyze results for insights
    task_accuracies = {}
    model_performance = {}
    eval_type_comparison = {}
    
    for result in results:
        # Task accuracies
        if result.task not in task_accuracies:
            task_accuracies[result.task] = []
        task_accuracies[result.task].append(result.accuracy)
        
        # Model performance
        if result.model_size not in model_performance:
            model_performance[result.model_size] = []
        model_performance[result.model_size].append(result.accuracy)
        
        # Eval type comparison
        if result.eval_type not in eval_type_comparison:
            eval_type_comparison[result.eval_type] = []
        eval_type_comparison[result.eval_type].append(result.accuracy)
    
    # Write insights
    with insights_file.open("w") as f:
        f.write("# Natural-Plan Evaluation Analysis Insights\n\n")
        f.write(f"Generated from {len(results)} evaluation results\n\n")
        
        f.write("## Task Performance Summary\n\n")
        for task, accuracies in task_accuracies.items():
            avg_acc = sum(accuracies) / len(accuracies)
            f.write(f"- **{task.title()}**: {avg_acc:.3f} average accuracy ({len(accuracies)} evaluations)\n")
        
        f.write("\n## Model Size Performance\n\n")
        for model, accuracies in model_performance.items():
            avg_acc = sum(accuracies) / len(accuracies)
            f.write(f"- **{model}**: {avg_acc:.3f} average accuracy ({len(accuracies)} evaluations)\n")
        
        f.write("\n## Evaluation Type Comparison\n\n")
        for eval_type, accuracies in eval_type_comparison.items():
            avg_acc = sum(accuracies) / len(accuracies)
            f.write(f"- **{eval_type}**: {avg_acc:.3f} average accuracy ({len(accuracies)} evaluations)\n")
        
        # Token efficiency analysis
        f.write("\n## Token Efficiency Insights\n\n")
        budget_results = [r for r in results if r.eval_type == "budget"]
        baseline_results = [r for r in results if r.eval_type == "baseline"]
        
        if budget_results and baseline_results:
            avg_budget_tokens = sum(r.avg_tokens_per_question for r in budget_results) / len(budget_results)
            avg_baseline_tokens = sum(r.avg_tokens_per_question for r in baseline_results) / len(baseline_results)
            token_reduction = (avg_baseline_tokens - avg_budget_tokens) / avg_baseline_tokens * 100
            
            f.write(f"- Budget evaluations use {avg_budget_tokens:.1f} tokens/question on average\n")
            f.write(f"- Baseline evaluations use {avg_baseline_tokens:.1f} tokens/question on average\n")
            f.write(f"- Token reduction: {token_reduction:.1f}%\n")
    
    print(f"* Analysis insights saved to: {insights_file}")
    
    # 6. Final summary
    print("\n✓ Batch analysis complete!")
    print("=" * 60)
    print(f"* All outputs saved to: {output_dir}")
    print(f"* Summary report: {output_dir}/evaluation_summary.csv")
    if not args.skip_plots:
        print(f"* Plots directory: {plots_dir}")
    print(f"* Insights: {insights_file}")
    
    if not args.skip_missing_check and missing_baseline:
        print(f"\n! Don't forget to run missing evaluations!")
        print(f"   Commands saved in: {output_dir}/missing_eval_commands.sh")


def _run_budget_comparison(results_dir: Path, output_dir: Path):
    """Run budget comparison analysis only."""
    print("- Running budget comparison analysis...")
    budget_results_dir = Path(PATHS["budget_results_dir"])
    if not budget_results_dir.exists():
        print(f"✗ Budget results directory not found: {budget_results_dir}")
        return
    
    output_csv = output_dir / "budget_comparison.csv"
    scan_budget_results(budget_results_dir, output_csv, rerun=False, clean=False)
    print(f"* Budget comparison complete: {output_csv}")


def _run_plotting_only(results_dir: Path, output_dir: Path, task: str = None, models: list = None):
    """Run plotting analysis only."""
    print("- Running plotting analysis...")
    analyzer = ResultsAnalyzer(results_dir)
    results = analyzer.scan_all_results()
    
    if not results:
        print("✗ No results found to plot!")
        return
    
    dirs = ensure_output_dirs(output_dir)
    plots_dir = dirs["plots_dir"]
    
    if task:
        plot_accuracy(results, task=task, models=models, output_dir=plots_dir)
    else:
        # Generate plots for all tasks
        for task_name in TASKS:
            plot_accuracy(results, task=task_name, models=models, output_dir=plots_dir)
        # Combined plot
        plot_accuracy(results, task=None, models=models, output_dir=plots_dir)
    
    print(f"* Plotting complete: {plots_dir}")


def _run_summary_only(results_dir: Path, output_dir: Path):
    """Run summary analysis only."""
    print("- Running summary analysis...")
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.scan_all_results()
    analyzer.generate_summary_report(output_dir)
    print(f"* Summary complete: {output_dir}")


if __name__ == "__main__":
    main()


