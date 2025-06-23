"""
Plot output tokens vs. decode latency for all models (server side only).
Each model gets its own figure.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from configs.qwen_1_5b import DATA_CONFIG as QWEN_1_5B_DATA
from configs.llama_8b import DATA_CONFIG as LLAMA_8B_DATA
from configs.qwen_14b import DATA_CONFIG as QWEN_14B_DATA

MODELS = [
    ("Qwen-1.5B", QWEN_1_5B_DATA, "DeepSeek-R1-Distill-Qwen-1_5B"),
    ("LLaMA-8B", LLAMA_8B_DATA, "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen-14B", QWEN_14B_DATA, "DeepSeek-R1-Distill-Qwen-14B"),
]


def plot_server_decode(model_name, data_config, server_sheet, save_dir):
    server_df = pd.read_excel(data_config.server_data_path, sheet_name=server_sheet)
    n_server = len(server_df)
    plt.figure(figsize=(7, 5))
    plt.scatter(server_df["output_tokens"], server_df["decode_time"] / 1000.0, alpha=0.4, label=f"Server (n={n_server})", s=10)
    plt.xlabel("Output Tokens (Decode)")
    plt.ylabel("Decode Latency (s)")
    plt.title(f"{model_name}: Output Tokens vs. Decode Latency (Server)")
    plt.legend()
    plt.tight_layout()
    out_path = Path(save_dir) / f"{model_name.replace(' ', '_').lower()}_server_decode.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved decode plot for {model_name} to {out_path}")


def main():
    save_dir = Path("outputs/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    for model_name, data_config, server_sheet in MODELS:
        plot_server_decode(model_name, data_config, server_sheet, save_dir)

if __name__ == "__main__":
    main()
