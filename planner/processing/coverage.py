#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Results scanner and coverage checker."""

import argparse
from pathlib import Path
import re
from typing import List, Tuple
from ..config import PATHS


class ScanResults:
    """Scans and analyzes evaluation results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        
    def scan_all_evaluations(self) -> List[Tuple[str, str, str]]:
        """Return (task, model, eval_type) tuples from results directory."""
        found_results = []
        
        # Scan base/ directory for baseline results
        base_dir = Path(PATHS["baseline_results_dir"])
        if base_dir.exists():
            for model_dir in base_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                for task_dir in model_dir.iterdir():
                    if not task_dir.is_dir():
                        continue
                    
                    match = re.match(r'([^_]+)_\d{8}_\d{6}', task_dir.name)
                    if match:
                        task = match.group(1)
                        json_files = list(task_dir.glob("results_*.json"))
                        if json_files:
                            found_results.append((task, model, "baseline"))
        
        budget_dir = Path(PATHS["budget_results_dir"])
        if budget_dir.exists():
            for model_dir in budget_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                self._scan_budget_recursive(model_dir, model, found_results)
        
        return found_results
    
    def _scan_budget_recursive(self, search_dir: Path, model: str, found_results: List, token_budget: str = None):
        for subdir in search_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            if subdir.name.isdigit():
                self._scan_budget_recursive(subdir, model, found_results, subdir.name)
            else:
                # This should be a task directory
                match = re.match(r'([^_]+)_\d{8}_\d{6}', subdir.name)
                if match:
                    task = match.group(1)
                    # Check if results file exists
                    json_files = list(subdir.glob("results_*.json"))
                    if json_files:
                        eval_type = f"budget-{token_budget}" if token_budget else "budget"
                        found_results.append((task, model, eval_type))
        
    def find_missing_baseline_evals(self, required_tasks: List[str] = None, 
                                  required_models: List[str] = None) -> List[Tuple[str, str]]:
        if required_tasks is None:
            required_tasks = ["meeting", "calendar", "trip"]
        if required_models is None:
            required_models = ["14b", "8b", "1.5b"]
            
        existing_results = self.scan_all_evaluations()
        
        existing_baseline = set()
        for task, model, eval_type in existing_results:
            if eval_type == "baseline":
                existing_baseline.add((task, model))
        
        missing = []
        for task in required_tasks:
            for model in required_models:
                if (task, model) not in existing_baseline:
                    missing.append((task, model))
        
        return missing
    
    def find_baseline_results(self, tasks: List[str] = None, models: List[str] = None):
        all_results = self.scan_all_evaluations()
        
        baseline_results = []
        for task, model, eval_type in all_results:
            if eval_type != "baseline":
                continue
            if tasks and task not in tasks:
                continue
            if models and model not in models:
                continue
                
            baseline_dir = Path(PATHS["baseline_results_dir"])
            model_dir = baseline_dir / model
            
            task_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith(f"{task}_")]
            if not task_dirs:
                continue
                
            task_dir = task_dirs[0]
            json_files = list(task_dir.glob("results_*.json"))
            if not json_files:
                continue
                
            match = re.match(r'([^_]+)_(\d{8}_\d{6})', task_dir.name)
            timestamp = match.groups()[1] if match else "unknown"
            
            result = type('Result', (), {
                'task': task,
                'model_size': model,
                'timestamp': timestamp,
                'result_path': json_files[0]
            })()
            baseline_results.append(result)
        
        return baseline_results

    def check_budget_coverage(self) -> dict:
        existing_results = self.scan_all_evaluations()
        
        baseline_combos = set()
        budget_combos = set()
        token_budgets = {}
        
        for task, model, eval_type in existing_results:
            combo = (task, model)
            if eval_type == "baseline":
                baseline_combos.add(combo)
            elif eval_type.startswith("budget"):
                budget_combos.add(combo)
                if combo not in token_budgets:
                    token_budgets[combo] = set()
                if "-" in eval_type:
                    token_budget = eval_type.split("-")[1]
                    token_budgets[combo].add(token_budget)
        
        return {
            "baseline_only": baseline_combos - budget_combos,
            "budget_only": budget_combos - baseline_combos,
            "both": baseline_combos & budget_combos,
            "token_budgets": token_budgets
        }
    
    def get_found_evaluations(self) -> dict:
        existing_results = self.scan_all_evaluations()
        
        summary = {}
        
        for task, model, eval_type in existing_results:
            if eval_type not in summary:
                summary[eval_type] = []
            summary[eval_type].append(f"{task} ({model})")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Check Natural-Plan evaluation coverage")
    parser.add_argument("--results-dir", default="data/planner/",
                       help="Directory containing evaluation results")
    parser.add_argument("--task", help="Check specific task only")
    parser.add_argument("--models", help="Comma-separated list of models (14b,8b,1.5b)")
    parser.add_argument("--debug", action="store_true",
                       help="Show detailed info about found evaluations")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    required_tasks = [args.task] if args.task else ["meeting", "calendar", "trip"]
    required_models = args.models.split(",") if args.models else ["14b", "8b", "1.5b"]
    
    checker = ScanResults(results_dir)
    
    if args.debug:
        found_evals = checker.get_found_evaluations()
        print("FOUND EVALUATIONS:")
        print("=" * 30)
        
        if "baseline" in found_evals:
            print(f"Baseline evaluations: {len(found_evals['baseline'])}")
            for eval_info in sorted(found_evals['baseline']):
                print(f"  {eval_info}")
            print()
        
        # Show budget evaluations by token budget
        budget_types = [k for k in found_evals.keys() if k.startswith("budget")]
        for budget_type in sorted(budget_types):
            token_budget = budget_type.split("-")[1] if "-" in budget_type else "unknown"
            print(f"Budget-{token_budget} evaluations: {len(found_evals[budget_type])}")
            for eval_info in sorted(found_evals[budget_type]):
                print(f"  {eval_info}")
            print()
    
    # Find missing baseline evaluations
    missing_baseline = checker.find_missing_baseline_evals(required_tasks, required_models)
    
    if missing_baseline:
        print("MISSING BASELINE EVALUATIONS:")
        print("=" * 50)
        for task, model in missing_baseline:
            print(f"  {task} task with {model} model")
        print()
    else:
        print("All required baseline evaluations found!")
    
    coverage = checker.check_budget_coverage()
    
    print("EVALUATION COVERAGE SUMMARY:")
    print("=" * 35)
    total_baseline = len(coverage['both']) + len(coverage['baseline_only'])
    total_budget = len(coverage['both']) + len(coverage['budget_only']) 
    print(f"  Baseline evaluations found: {total_baseline}")
    print(f"  Budget evaluations found: {total_budget}")
    print(f"  Complete pairs (both types): {len(coverage['both'])}")
    
    if coverage['token_budgets']:
        token_counts = {}
        for combo, tokens in coverage['token_budgets'].items():
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        print(f"\n  Token budget breakdown:")
        for token in sorted(token_counts.keys(), key=int):
            print(f"    {token} tokens: {token_counts[token]} evaluations")
    
    if coverage['baseline_only']:
        print(f"\nBaseline-only (missing budget): {len(coverage['baseline_only'])}")
        for task, model in sorted(coverage['baseline_only']):
            print(f"    {task} ({model})")
    
    if coverage['budget_only']:
        print(f"\nBudget-only (missing baseline): {len(coverage['budget_only'])}")
        for task, model in sorted(coverage['budget_only']):
            print(f"    {task} ({model})")
    
    if len(coverage['both']) > 0 and not coverage['baseline_only'] and not coverage['budget_only']:
        print("\nStatus: Complete coverage! All evaluations have both baseline and budget results.")


if __name__ == "__main__":
    main()


