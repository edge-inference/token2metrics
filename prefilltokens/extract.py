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
Extract the synthetically generated points used for fitting and save to CSV.
These are for input lengths in the multiples of 128 tokens.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
script_dir = Path(__file__).parent.parent.resolve()
xlsx_candidates = list((script_dir / "data/synthetic/gpu/prefill/processed").glob("all_results_by_model*.xlsx"))
if not xlsx_candidates:
    raise FileNotFoundError(
        f"No matching XLSX file found in processed/ (searched in {script_dir})"
    )
JETSON_XLSX_PATH = xlsx_candidates[0]

MODELS = [
    ("Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1_5B"),
    ("LLaMA-8B",  "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen-14B",  "DeepSeek-R1-Distill-Qwen-14B"),
]

MODEL_RELEASE_NAMES = {
    "Qwen-1.5B": "DSR1-Qwen-1.5B",
    "LLaMA-8B": "DSR1-LLaMA-8B",
    "Qwen-14B": "DSR1-Qwen-14B"
}

PLOT_TARGETS = [64, 128, 256, 384, 512, 1024]


def select_closest_points(tokens: np.ndarray, prefill: np.ndarray, targets: list, ignore_384: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Select the closest points to target token values."""
    selected_tokens = []
    selected_prefill = []
    
    for target in targets:
        if ignore_384 and target == 384:
            continue
        idx = np.argmin(np.abs(tokens - target))
        selected_tokens.append(tokens[idx])
        selected_prefill.append(prefill[idx])
    
    return np.array(selected_tokens), np.array(selected_prefill)


def extract_fitting_points() -> pd.DataFrame:
    """Extract fitting points for all models and return as DataFrame."""
    all_points = []
    xls = pd.ExcelFile(JETSON_XLSX_PATH)
    
    for model_name, sheet_name in MODELS:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"[ERROR] Could not read sheet '{sheet_name}': {e}")
            continue
            
        if 'ttft' not in df.columns:
            print(f"[WARNING] No 'ttft' column for {model_name}, skipping.")
            continue

        tokens = df["input_tokens"].values
        prefill = df["ttft"].values / 1000.0  # Convert to seconds
        
        # Apply same filtering as in main.py
        mask = tokens <= 650
        tokens_filtered = tokens[mask]
        prefill_filtered = prefill[mask]

        # Select closest points to targets
        ignore_384_for_qwen14b = (model_name == "Qwen-14B")
        selected_tokens, selected_prefill = select_closest_points(
            tokens_filtered, prefill_filtered, PLOT_TARGETS, ignore_384=ignore_384_for_qwen14b
        )
        
        # Create records for this model
        model_display_name = MODEL_RELEASE_NAMES.get(model_name, model_name)
        
        for token_val, prefill_val in zip(selected_tokens, selected_prefill):
            all_points.append({
                'model': model_display_name,
                'model_internal': model_name,
                'input_tokens': int(token_val),
                'prefill_latency_seconds': float(prefill_val)
            })
    
    return pd.DataFrame(all_points)


def main():
    """Extract fitting points and save to CSV."""
    print(f"Extracting fitting points from: {JETSON_XLSX_PATH}")
    
    # Extract points
    fitting_points_df = extract_fitting_points()
    
    # Save to CSV
    output_path = Path("plots") / "fitting_points.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    fitting_points_df.to_csv(output_path, index=False)
    
    print(f"\nExtracted {len(fitting_points_df)} fitting points")
    print(f"Saved to: {output_path}")
    
    # Display summary
    print("\nSummary by model:")
    summary = fitting_points_df.groupby('model').agg({
        'input_tokens': ['count', 'min', 'max'],
        'prefill_latency_seconds': ['min', 'max']
    }).round(4)
    print(summary)
    
    # Display first few rows
    print(f"\nFirst few rows:")
    print(fitting_points_df.head(10))


if __name__ == "__main__":
    main()
