#!/usr/bin/env python3
"""
Run static evaluations using Google's original evaluation scripts.

This script automates the complete workflow:
1. Find baseline evaluation results
2. Create evaluation datasets by replacing pred_5shot_pro with model predictions  
3. Run Google's evaluation scripts
4. Save results for analysis

Usage:
    python processor/run_static_eval.py
    python processor/run_static_eval.py --task meeting --model 14b
    python processor/run_static_eval.py --results-dir custom_results/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
from .merge import EvalDatasetMerger
from .coverage import ScanResults
from ..config import PATHS, EVAL_SCRIPTS, ensure_directories



class EvalScorer:
    """Scores evaluations using Google's evaluation scripts"""
    
    def __init__(self, results_dir: Path, eval_data_dir: Path = None, output_dir: Path = None):
        self.results_dir = results_dir
        self.eval_data_dir = eval_data_dir or Path(PATHS["reference_data"])
        self.output_dir = output_dir or Path("processor/static_eval_results")
        self.merger = EvalDatasetMerger(self.eval_data_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_baseline_results(self, tasks: List[str] = None, models: List[str] = None):
        """Find all baseline evaluation results"""
        scanner = ScanResults(self.results_dir)
        return scanner.find_baseline_results(tasks, models)
    
    def run_static_evaluation(self, baseline_result, verbose: bool = True) -> Dict[str, Any]:
        """Run static evaluation for a single baseline result"""
        task = baseline_result.task
        model = baseline_result.model_size
        timestamp = baseline_result.timestamp
        
        if verbose:
            print(f"\n* Running static evaluation: {task} task, {model} model")
            print(f"   Source: {baseline_result.result_path}")
        
        try:
            eval_dataset_name = f"{task}_{model}_{timestamp}_eval.json"
            eval_dataset_path = self.output_dir / "datasets" / eval_dataset_name
            eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                print(f"- Creating evaluation dataset...")
            
            self.merger.merge_predictions(
                baseline_result.result_path, 
                task, 
                eval_dataset_path
            )
            
            # Run evaluation
            if verbose:
                print(f"- Running Google evaluation script...")
            
            eval_result = self.merger.run_evaluation(eval_dataset_path, task)
            
            # Add metadata
            result = {
                "task": task,
                "model_size": model,
                "timestamp": timestamp,
                "accuracy": eval_result["accuracy"],
                "eval_dataset_path": str(eval_dataset_path),
                "original_result_path": str(baseline_result.result_path),
                "evaluation_output": eval_result["raw_output"],
                "eval_script": eval_result["eval_script"]
            }
            
            if verbose:
                print(f"✓ Static evaluation complete: {result['accuracy']:.4f} accuracy")
            
            return result
            
        except Exception as e:
            error_result = {
                "task": task,
                "model_size": model,
                "timestamp": timestamp,
                "accuracy": 0.0,
                "error": str(e),
                "original_result_path": str(baseline_result.result_path)
            }
            
            if verbose:
                print(f"✗ Static evaluation failed: {e}")
            
            return error_result
    
    def run_all_evaluations(self, tasks: List[str] = None, models: List[str] = None,
                                 verbose: bool = True) -> List[Dict[str, Any]]:
        """Run static evaluations for all found baseline results"""
        
        if verbose:
            print("- Finding baseline evaluation results...")
        
        baseline_results = self.find_baseline_results(tasks, models)
        
        static_eval_results = []
        
        if not baseline_results and not static_eval_results:
            print("✗ No baseline results found!")
            print("   Make sure you have run baseline evaluations first.")
            print("   Use: python processor/check_coverage.py")
            return []
        
        if verbose:
            print(f"* Found {len(baseline_results)} baseline results to evaluate")
        
        for baseline_result in baseline_results:
            result = self.run_static_evaluation(baseline_result, verbose)
            static_eval_results.append(result)
        
        # Save all results
        summary_file = self.output_dir / "evaluation_summary.json"
        with summary_file.open("w") as f:
            json.dump(static_eval_results, f, indent=2)
        
        if verbose:
            print(f"\n* EVALUATION SUMMARY")
            print("=" * 50)
            print(f"{'Task':<10} {'Model':<6} {'Accuracy':<10} {'Status'}")
            print("-" * 50)
            
            for result in static_eval_results:
                status = "✓ OK" if "error" not in result else "✗ Error"
                print(f"{result['task']:<10} {result['model_size']:<6} {result['accuracy']:<10.4f} {status}")
            
            print(f"\n* Full results saved: {summary_file}")
        
        return static_eval_results
    
    def create_detailed_csv_report(self, static_results: List[Dict[str, Any]]) -> None:
        """Create a detailed CSV report from the static evaluation results."""
        
        report_file = self.output_dir / "detailed_evaluation_summary.csv"
        
        with report_file.open("w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["model", "task", "accuracy", "avg_time_ms", "avg_prompt_tokens", "avg_tokens_per_q", "avg_tokens_per_second"])
            
            for result in static_results:
                if "error" in result:
                    writer.writerow([result["model_size"], result["task"], 0.0, 0, 0, 0, 0.0])
                    continue
                
                model = result["model_size"]
                task = result["task"]
                accuracy = result["accuracy"]
                
                performance_metrics = self._extract_performance_metrics(result.get("original_result_path"))
                
                writer.writerow([
                    model,
                    task, 
                    accuracy,
                    performance_metrics.get("avg_time_ms", 0),
                    performance_metrics.get("avg_prompt_tokens", 0),
                    performance_metrics.get("avg_tokens_per_q", 0),
                    performance_metrics.get("avg_tokens_per_second", 0.0)
                ])

        print(f"* Detailed CSV report saved: {report_file}")

    def _extract_performance_metrics(self, result_path: str) -> Dict[str, float]:
        """Extract performance metrics from original result JSON file."""
        metrics = {}
        
        if not result_path:
            return metrics
            
        try:
            result_file = Path(result_path)
            if not result_file.exists():
                return metrics
                
            with result_file.open() as f:
                data = json.load(f)
            
            # Extract metrics from question_results
            qrs = data.get("question_results", [])
            if qrs:
                # Calculate averages
                total_time_ms = sum(q.get("time_ms", 0.0) for q in qrs)
                total_prompt_tokens = sum(q.get("prompt_tokens", 0) for q in qrs)
                total_output_tokens = sum(q.get("output_tokens", 0) for q in qrs)
                total_tokens_per_second = sum(q.get("tokens_per_second", 0.0) for q in qrs)
                
                num_questions = len(qrs)
                metrics["avg_time_ms"] = round(total_time_ms / num_questions)
                metrics["avg_prompt_tokens"] = round(total_prompt_tokens / num_questions)
                metrics["avg_tokens_per_q"] = round(total_output_tokens / num_questions)
                metrics["avg_tokens_per_second"] = round(total_tokens_per_second / num_questions, 2)
                
        except Exception as e:
            pass
            
        return metrics

    def _parse_detailed_metrics(self, raw_output: str, task: str) -> Dict[str, float]:
        """Parse detailed metrics from the raw output of an evaluation script."""
        metrics = {}
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                parts = line.split(":")
                key = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    if value > 1.0:
                        continue
                    key = key.replace(" of 1000 samples", "").replace(" of 100 samples", "")
                    key = key.replace("Overall solve rate", "accuracy_total")
                    key = key.replace("Accuracy for all", "accuracy_total")
                    key = key.replace("Solve rate of ", "solve_rate_")
                    key = key.replace(" ", "_").replace(",", "")
                    metrics[key] = value
                except (ValueError, IndexError):
                    continue
        
        return metrics

    def create_comparison_report(self, static_results: List[Dict[str, Any]]) -> None:
        """DEPRECATED: This method is no longer used in favor of the detailed CSV report."""
        print("* Markdown comparison report is deprecated and will not be generated.")
        pass


def main():
    parser = argparse.ArgumentParser(description="Run static evaluations using Google's evaluation scripts")
    parser.add_argument("--results-dir", default="results/",
                       help="Directory containing evaluation results")
    parser.add_argument("--eval-data-dir", default=PATHS["reference_data"],
                       help="Directory containing original evaluation datasets")
    parser.add_argument("--output-dir", default="processor/static_eval_results",
                       help="Directory to save static evaluation results")
    parser.add_argument("--task", choices=["meeting", "calendar", "trip"],
                       help="Run evaluation for specific task only")
    parser.add_argument("--model", choices=["14b", "8b", "1.5b"],
                       help="Run evaluation for specific model only")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    eval_data_dir = Path(args.eval_data_dir)
    output_dir = Path(args.output_dir)
    
    ensure_directories()
    
    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        return 1
    
    missing_scripts = [script for script in EVAL_SCRIPTS.values() if not Path(script).exists()]
    if missing_scripts:
        print("✗ Missing evaluation scripts:")
        for script in missing_scripts:
            print(f"   {script}")
        return 1
    
    tasks = [args.task] if args.task else None
    models = [args.model] if args.model else None
    verbose = not args.quiet
    
    # Run evaluations
    runner = EvalScorer(results_dir, eval_data_dir, output_dir)
    static_results = runner.run_all_evaluations(tasks, models, verbose)
    
    if not static_results:
        return 1
    
    # Create detailed CSV report
    if verbose:
        print("\n* Creating detailed CSV report...")
    runner.create_detailed_csv_report(static_results)

    if verbose:
        print(f"\n✓ Evaluation workflow complete!")
    print(f"* All outputs saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())