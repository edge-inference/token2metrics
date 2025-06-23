"""
Plot input tokens vs. prefill latency for all models (server and Jetson) on a single plot.
Includes L1MAX. Each model is a different color/label.
"""
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path for config imports
sys.path.append(str(Path(__file__).parent.parent))

from configs.qwen_1_5b import DATA_CONFIG as QWEN_1_5B_DATA
from configs.llama_8b import DATA_CONFIG as LLAMA_8B_DATA
from configs.qwen_14b import DATA_CONFIG as QWEN_14B_DATA
from configs.l1max_qwen_1_5b import DATA_CONFIG as L1MAX_DATA

MODELS = [
    ("L1MAX", L1MAX_DATA, "L1-Qwen-1_5B-Max", "profiling_combined_L1Max.csv"),
    ("Qwen-1.5B", QWEN_1_5B_DATA, "DeepSeek-R1-Distill-Qwen-1_5B", "profiling_combined_DSR1-Qwen-1.5B.csv"),
    ("LLaMA-8B", LLAMA_8B_DATA, "DeepSeek-R1-Distill-Llama-8B", "profiling_combined_DSR1-LLama-8B.csv"),
    ("Qwen-14B", QWEN_14B_DATA, "DeepSeek-R1-Distill-Qwen-14B", "profiling_combined_DSR1-Qwen-14B.csv"),
]

JETSON_XLSX_PATH = Path("datasets/tegra/full_mmlu_by_model_tegra.xlsx")


# Mapping from short model name to full release name for legend
MODEL_RELEASE_NAMES = {
    "L1MAX": "L1-Qwen-1.5B-Max",
    "Qwen-1.5B": "DeepSeek-R1-Distill-Qwen-1_5B",
    "LLaMA-8B": "DeepSeek-R1-Distill-Llama-8B",
    "Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B"
}

def main():
    plt.figure(figsize=(8, 6))
    for model_name, data_config, server_sheet, jetson_csv in MODELS:
        # Load Jetson data: prefer Excel if available, else fallback to CSV
        if JETSON_XLSX_PATH.exists():
            try:
                jetson_df = pd.read_excel(JETSON_XLSX_PATH, sheet_name=server_sheet)
            except ValueError:
                print(f"[WARN] Jetson Excel: Sheet '{server_sheet}' not found, skipping {model_name}.")
                continue
            jetson_prefill = jetson_df["ttft"] / 1000.0
            jetson_input_tokens = jetson_df["input_tokens"]
        else:
            jetson_path = Path(data_config.jetson_data_path) / jetson_csv
            jetson_df = pd.read_csv(jetson_path, encoding='latin-1')
            jetson_prefill = jetson_df["prefill"] / 1000.0
            jetson_input_tokens = jetson_df["output_tokens"]
        n_jetson = len(jetson_input_tokens)
        legend_name = f"{MODEL_RELEASE_NAMES.get(model_name, model_name)} (n={n_jetson})"
        plt.scatter(jetson_input_tokens, jetson_prefill, alpha=0.5, s=16, label=legend_name)
    plt.xlabel("Input Tokens")
    plt.ylabel("Prefill Latency (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path("outputs/plots/prefill_vs_tokens_all_models.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved prefill vs. input tokens plot for all models to {out_path}")

if __name__ == "__main__":
    main()
