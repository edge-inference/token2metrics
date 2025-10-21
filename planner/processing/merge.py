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

"""Merge model predictions into evaluation datasets."""

import json
from pathlib import Path
from typing import Dict, Any, List
import shutil
from ..config import EVAL_SCRIPTS, EVAL_DATA_MAP, PATHS


class EvalDatasetMerger:
    
    def __init__(self, eval_data_dir: Path = None):
        self.eval_data_dir = eval_data_dir or Path(PATHS["reference_data"])
        
    def merge_predictions(self, results_json_path: Path, task: str, output_path: Path = None) -> Path:
        original_dataset_path = Path(EVAL_DATA_MAP[task])
        
        if not original_dataset_path.exists():
            raise FileNotFoundError(f"Original dataset not found: {original_dataset_path}")
        
        print(f"- Loading original dataset: {original_dataset_path}")
        with original_dataset_path.open() as f:
            original_data = json.load(f)
        
        # Load model results
        print(f"- Loading model results: {results_json_path}")
        with results_json_path.open() as f:
            results_data = json.load(f)
        
        # Create mapping of question_id -> generated_text
        predictions = {}
        for result in results_data.get("question_results", []):
            question_id = str(result.get("question_id", ""))
            generated_text = result.get("generated_text", "")
            if question_id and generated_text:
                predictions[question_id] = generated_text
        
        print(f"- Found {len(predictions)} predictions to merge")
        
        # Clone dataset and replace predictions
        eval_dataset = {}
        matched_count = 0
        
        for qid, item in original_data.items():
            eval_item = dict(item)
            
            if qid in predictions:
                eval_item["pred_5shot_pro"] = predictions[qid]
                matched_count += 1
            else:
                print(f"! No prediction found for question {qid}, keeping original")
            eval_dataset[qid] = eval_item

        print(f"✓ Replaced {matched_count}/{len(original_data)} predictions")
        if output_path is None:
            parts = results_json_path.parts
            model_info = "unknown"
            timestamp = "unknown"
            
            for part in parts:
                if part in ["14b", "8b", "1.5b"]:
                    model_info = part
                if "_" in part and len(part) == 15:  # YYYYMMDD_HHMMSS format
                    timestamp = part
            
            output_path = Path(PATHS["merged_results_dir"]) / f"{task}_{model_info}_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as f:
            json.dump(eval_dataset, f, indent=2)
        
        print(f"* Evaluation dataset saved: {output_path}")
        return output_path
    
    def run_evaluation(self, eval_dataset_path: Path, task: str) -> Dict[str, Any]:
        import subprocess
        import sys
        
        eval_script = EVAL_SCRIPTS.get(task)
        if not eval_script or not Path(eval_script).exists():
            raise FileNotFoundError(f"Evaluation script not found: {eval_script}")
        
        print(f"- Running evaluation script: {eval_script}")
        
        # Run evaluation script
        try:
            result = subprocess.run([
                sys.executable, eval_script,
                f"--data_path={eval_dataset_path}"
            ], capture_output=True, text=True, check=True)
            
            output = result.stdout
            print("- Evaluation output:")
            print(output)
            
            accuracy = self._parse_accuracy_from_output(output, task)
            
            return {
                "accuracy": accuracy,
                "raw_output": output,
                "eval_script": eval_script,
                "dataset_path": str(eval_dataset_path)
            }
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Evaluation failed: {e}")
            print(f"Error output: {e.stderr}")
            raise
    
    def _parse_accuracy_from_output(self, output: str, task: str) -> float:
        """Parse accuracy from evaluation script output"""
        lines = output.strip().split('\n')
        
        if task == "meeting":
            # Look for "Accuracy for all: X.XXX"
            for line in lines:
                if "Accuracy for all:" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
        
        elif task == "calendar":
            # Look for "Overall solve rate of X samples: X.XXX"
            for line in lines:
                if "Overall solve rate" in line and "samples:" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
        
        elif task == "trip":
            # Look for "EM Accuracy of X samples: X.XXX"
            for line in lines:
                if "EM Accuracy" in line and "samples:" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
        
        print(f"! Could not parse accuracy from output, returning 0.0")
        return 0.0
    






