"""
Plot input/output tokens vs. prefill/decode latency for all models (Jetson only).
Each model gets a figure with two subplots: prefill and decode.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path for config imports
sys.path.append(str(Path(__file__).parent.parent))

from configs.qwen_1_5b import DATA_CONFIG as QWEN_1_5B_DATA
from configs.llama_8b import DATA_CONFIG as LLAMA_8B_DATA
from configs.qwen_14b import DATA_CONFIG as QWEN_14B_DATA

# Model info: (name, data config, jetson sheet)
MODELS = [
    ("Qwen-1.5B", QWEN_1_5B_DATA, "DeepSeek-R1-Distill-Qwen-1_5B"),
    ("LLaMA-8B", LLAMA_8B_DATA, "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen-14B", QWEN_14B_DATA, "DeepSeek-R1-Distill-Qwen-14B"),
]

JETSON_XLSX_PATH = Path("datasets/tegra/full_mmlu_by_model_tegra.xlsx")


def plot_model_token_latency(model_name, jetson_sheet, save_dir):
    if not JETSON_XLSX_PATH.exists():
        print(f"[ERROR] Jetson Excel file not found at {JETSON_XLSX_PATH}")
        return
    try:
        jetson_df = pd.read_excel(JETSON_XLSX_PATH, sheet_name=jetson_sheet)
    except ValueError:
        print(f"[WARN] Jetson Excel: Sheet '{jetson_sheet}' not found, skipping {model_name}.")
        return
    jetson_prefill = jetson_df["ttft"] / 1000.0
    jetson_decode = jetson_df["decode_time"] / 1000.0
    jetson_input_tokens = jetson_df["input_tokens"]
    jetson_output_tokens = jetson_df["output_tokens"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name}: Token Count vs. Latency (Jetson)")

    # Prefill: input_tokens vs. prefill (ttft)
    n_jetson = len(jetson_df)
    axes[0].scatter(jetson_input_tokens, jetson_prefill, alpha=0.6, label=f"Jetson (n={n_jetson})", s=16)
    axes[0].set_xlabel("Input Tokens (Prefill)")
    axes[0].set_ylabel("Prefill Latency (s)")
    axes[0].set_title("Prefill Phase")
    axes[0].legend()

    # Decode: output_tokens vs. decode_time
    axes[1].scatter(jetson_output_tokens, jetson_decode, alpha=0.6, label=f"Jetson (n={n_jetson})", s=16)
    axes[1].set_xlabel("Output Tokens (Decode)")
    axes[1].set_ylabel("Decode Latency (s)")
    axes[1].set_title("Decode Phase")
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = Path(save_dir) / f"{model_name.replace(' ', '_').lower()}_token_latency.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot for {model_name} to {out_path}")


def main():
    save_dir = Path("outputs/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    for model_name, _, jetson_sheet in MODELS:
        plot_model_token_latency(model_name, jetson_sheet, save_dir)

if __name__ == "__main__":
    main()
