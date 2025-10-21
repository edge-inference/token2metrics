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

"""
Comprehensive Natural-Plan Results Analyzer

Analyzes evaluation results across baseline, budget, and scaling experiments for both reasoning and direct mode
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from .config import PATHS


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    task: str
    model_size: str
    eval_type: str  # baseline, budget, scaling, direct
    token_limit: Optional[int]
    accuracy: float
    avg_tokens_per_question: float
    total_questions: int
    avg_time_ms: float
    tokens_per_second: float
    timestamp: str
    result_path: Path


class ResultsAnalyzer:
    """Analyzes Natural-Plan evaluation results across different experiment types"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results: List[EvaluationResult] = []
        
    def scan_all_results(self) -> List[EvaluationResult]:
        """Scan all result directories and extract evaluation metrics"""
        print(f"- Scanning results in {self.results_dir}")
        
        self._scan_baseline_results()
        self._scan_budget_results()
        self._scan_scaling_results()
        
        print(f"* Found {len(self.results)} evaluation results")
        return self.results
    
    def _scan_baseline_results(self):
        """Scan base/ directory for baseline evaluations"""
        base_dir = Path(PATHS["baseline_results_dir"])
        if not base_dir.exists():
            return
            
        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_size = model_dir.name
            
            for task_dir in model_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                match = re.match(r'([^_]+)_(\d{8}_\d{6})', task_dir.name)
                if not match:
                    continue
                task, timestamp = match.groups()
                
                # Find results JSON file
                json_files = list(task_dir.glob("results_*.json"))
                if not json_files:
                    continue
                    
                result = self._parse_result_file(json_files[0], task, model_size, "baseline", None, timestamp)
                if result:
                    self.results.append(result)
    
    def _scan_budget_results(self):
        """Scan budget/ directory for budget-constrained evaluations"""
        budget_dir = Path(PATHS["budget_results_dir"])
        if not budget_dir.exists():
            return
            
        for model_dir in budget_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_size = model_dir.name
            
            for subdir in model_dir.iterdir():
                if not subdir.is_dir():
                    continue
                    
                if subdir.name.isdigit():
                    token_limit = int(subdir.name)
                    search_dir = subdir
                else:
                    token_limit = 256
                    search_dir = subdir
                    
                if any(d.is_dir() for d in search_dir.iterdir()):
                    for task_dir in search_dir.iterdir():
                        if not task_dir.is_dir():
                            continue
                        
                        match = re.match(r'([^_]+)_(\d{8}_\d{6})', task_dir.name)
                        if not match:
                            continue
                        task, timestamp = match.groups()
                        
                        json_files = list(task_dir.glob("results_*.json"))
                        if not json_files:
                            continue
                            
                        result = self._parse_result_file(json_files[0], task, model_size, "budget", token_limit, timestamp)
                        if result:
                            self.results.append(result)
                else:
                    match = re.match(r'([^_]+)_(\d{8}_\d{6})', search_dir.name)
                    if not match:
                        continue
                    task, timestamp = match.groups()
                    
                    json_files = list(search_dir.glob("results_*.json"))
                    if not json_files:
                        continue
                        
                    result = self._parse_result_file(json_files[0], task, model_size, "budget", token_limit, timestamp)
                    if result:
                        self.results.append(result)
    

    
    def _scan_scaling_results(self):
        """Scan scaling/ directory for scaling evaluations"""
        scaling_dir = Path(PATHS["scaling_results_dir"])
        if not scaling_dir.exists():
            return
        pass
    
    def _parse_result_file(self, json_path: Path, task: str, model_size: str, 
                          eval_type: str, token_limit: Optional[int], timestamp: str) -> Optional[EvaluationResult]:
        """Parse a results JSON file and extract metrics"""
        try:
            with json_path.open() as f:
                data = json.load(f)
            
            question_results = data.get("question_results", [])
            if not question_results:
                return None
                
            total_tokens = sum(q.get("prompt_tokens", 0) + q.get("output_tokens", 0) 
                             for q in question_results)
            avg_tokens = total_tokens / len(question_results) if question_results else 0
            
            return EvaluationResult(
                task=task,
                model_size=model_size,
                eval_type=eval_type,
                token_limit=token_limit,
                accuracy=data.get("accuracy", 0.0),
                avg_tokens_per_question=avg_tokens,
                total_questions=data.get("total_questions", len(question_results)),
                avg_time_ms=data.get("avg_time_per_question", 0.0),
                tokens_per_second=data.get("avg_tokens_per_second", 0.0),
                timestamp=timestamp,
                result_path=json_path
            )
            
        except Exception as e:
            print(f"! Error parsing {json_path}: {e}")
            return None
    

    
    def generate_summary_report(self, output_dir: Path = None):
        """Generate a comprehensive summary report"""
        if not self.results:
            print("✗ No results found. Run scan_all_results() first.")
            return
        
        df_data = []
        for result in self.results:
            df_data.append({
                'Task': result.task,
                'Model': result.model_size,
                'Eval Type': result.eval_type,
                'Token Limit': result.token_limit or 'None',
                'Accuracy': f"{result.accuracy:.4f}",
                'Avg Tokens/Q': f"{result.avg_tokens_per_question:.1f}",
                'Questions': result.total_questions,
                'Avg Time (ms)': f"{result.avg_time_ms:.1f}",
                'Tokens/sec': f"{result.tokens_per_second:.1f}",
                'Timestamp': result.timestamp
            })
        
        df = pd.DataFrame(df_data)
        
        # Print summary
        print("\n* EVALUATION SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        if output_dir:
            output_dir.mkdir(exist_ok=True)
            csv_path = output_dir / "evaluation_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n* Summary saved: {csv_path}")


