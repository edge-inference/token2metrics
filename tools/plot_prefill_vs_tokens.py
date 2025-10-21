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
Plot input tokens vs. prefill latency for all models 
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

JETSON_XLSX_PATH = Path("../../data/mmlu/gpu/full_mmlu_by_model_tegra.xlsx")

MODELS = [
    ("L1MAX",    "L1-Qwen-1_5B-Max"),
    ("Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1_5B"),
    ("LLaMA-8B",  "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen-14B",  "DeepSeek-R1-Distill-Qwen-14B"),
]

MODEL_RELEASE_NAMES = {
    "L1MAX":    "L1-Qwen-1.5B-Max",
    "Qwen-1.5B":"DeepSeek-R1-Distill-Qwen-1_5B",
    "LLaMA-8B": "DeepSeek-R1-Distill-Llama-8B",
    "Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B"
}

plt.rcParams.update({'font.size': 14})

def main():
    # 1) create figure+axis
    fig, ax = plt.subplots(figsize=(8, 6))
    xls = pd.ExcelFile(JETSON_XLSX_PATH)

    # 2) plot each model’s points
    for model_name, sheet_name in MODELS:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"[ERROR] Could not read sheet '{sheet_name}': {e}")
            continue
        if 'ttft' in df.columns:
            prefill = df["ttft"] / 1000.0
        else:
            print(f"[ERROR] No 'ttft' column found for {model_name}, skipping.")
            continue
        tokens = df["input_tokens"]

        label = f"{MODEL_RELEASE_NAMES.get(model_name, model_name)} (n={len(tokens)})"
        ax.scatter(tokens, prefill, s=16, alpha=0.5, label=label)

    # 3) labels & grid
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Prefill Latency (s)")
    ax.grid(True, alpha=0.3)

    # 4) pull handles+labels and create a figure-level legend below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        frameon=False
    )

    fig.subplots_adjust(bottom=0.15)

    out_path = Path("outputs/plots/prefill_vs_tokens_all_models.pdf")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved prefill vs. input Length plot for all models (Jetson only) to {out_path}")

if __name__ == "__main__":
    main()
