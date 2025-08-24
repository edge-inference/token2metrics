#!/usr/bin/env python3
"""Budget evaluation comparison utilities.

Compares performance across different token budget limits.
"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
from .processing.prune import clean_results

from .config import EVAL_DATA_MAP


def _prepare_results_json(orig: Path, clean: bool) -> Path:
    """Prepare results JSON, optionally cleaning predictions."""
    if not clean:
        return orig
    
    data = json.loads(orig.read_text())
    data = clean_results(data, data.get("subject", ""))
    
    tmp = Path(tempfile.mkstemp(suffix="_clean.json")[1])
    tmp.write_text(json.dumps(data))
    return tmp

def _run_google_eval(results_json: Path, task: str, clean: bool) -> float:
    """Run Google's eval script via EvalDatasetMerger class and return accuracy."""
    from .processing.merge import EvalDatasetMerger
    from .config import PATHS
    
    prepared_json = _prepare_results_json(results_json, clean)
    
    merger = EvalDatasetMerger(Path(PATHS["reference_data"]))
    
    with tempfile.NamedTemporaryFile(suffix="_eval.json", delete=False) as tmp:
        out_dataset = Path(tmp.name)
    
    try:
        merger.merge_predictions(prepared_json, task, out_dataset)
        
        eval_result = merger.run_evaluation(out_dataset, task)
        
        return eval_result["accuracy"]
        
    except Exception as e:
        print(f"! Evaluation failed: {e}")
        return 0.0
    finally:
        if prepared_json != results_json and prepared_json.exists():
            prepared_json.unlink()
        if out_dataset.exists():
            out_dataset.unlink()


def _scan_single_json(path: Path, task: str, rerun: bool, clean: bool) -> Tuple[float, float, float, float, float]:
    """Return (accuracy, avg_tokens_per_q, avg_time_ms, avg_prompt_tokens, avg_tokens_per_second) from results JSON."""
    with path.open() as f:
        data = json.load(f)
    acc = float(data.get("accuracy", 0.0))
    avg_tok = float(data.get("avg_tokens_per_q", 0.0))
    
    qrs = data.get("question_results", [])
    avg_time_ms = 0.0
    avg_prompt_tokens = 0.0
    avg_tokens_per_second = 0.0
    
    if qrs:
        total_time_ms = sum(q.get("time_ms", 0.0) for q in qrs)
        total_prompt_tokens = sum(q.get("prompt_tokens", 0) for q in qrs)
        total_tokens_per_second = sum(q.get("tokens_per_second", 0.0) for q in qrs)
        
        avg_time_ms = total_time_ms / len(qrs)
        avg_prompt_tokens = total_prompt_tokens / len(qrs)
        avg_tokens_per_second = total_tokens_per_second / len(qrs)
        
        if avg_tok == 0.0:
            try:
                tot = sum(q.get("output_tokens", 0) for q in qrs)
                avg_tok = tot / len(qrs)
            except Exception:
                pass
    
    if rerun:
        try:
            acc = _run_google_eval(path, task, clean)
        except Exception:
            pass
    
    return acc, avg_tok, avg_time_ms, avg_prompt_tokens, avg_tokens_per_second


def scan_budget_results(root: Path, out_csv: Path, args_rerun: bool, clean: bool) -> None:
    rows: List[List] = []

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name

        for budget_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            try:
                token_budget = int(budget_dir.name)
            except ValueError:
                continue

            for task_run in sorted(p for p in budget_dir.iterdir() if p.is_dir()):
                task = task_run.name.split("_")[0]

                json_files = list(task_run.glob("results_*.json"))
                if not json_files:
                    continue
                acc, avg_tok, avg_time_ms, avg_prompt_tokens, avg_tokens_per_second = _scan_single_json(json_files[0], task, args_rerun, clean)
                
                avg_tok_int = int(round(avg_tok))
                avg_time_ms_int = int(round(avg_time_ms))
                avg_prompt_tokens_int = int(round(avg_prompt_tokens))
                avg_tokens_per_second_rounded = round(avg_tokens_per_second, 2)
                
                rows.append([model, token_budget, task, acc, avg_tok_int, avg_time_ms_int, avg_prompt_tokens_int, avg_tokens_per_second_rounded])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "task", "token_budget", "accuracy", "avg_time_ms", "avg_prompt_tokens", "avg_tokens_per_q", "avg_tokens_per_second"])
        reordered_rows = [[row[0], row[2], row[1], row[3], row[5], row[6], row[4], row[7]] for row in rows]
        writer.writerows(reordered_rows)

    print(f"✓ Budget summary written → {out_csv}  (rows: {len(rows)})")



