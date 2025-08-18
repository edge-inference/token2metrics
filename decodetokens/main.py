"""
Plot decode latency vs. output tokens  Llama 8B model.
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

INPUT_TOKEN_LIST = [1, 128, 256, 511, 1013, 2048, 4096]

dir_path = Path(__file__).parent.parent / "datasets/decode"
pattern = re.compile(r"detailed_results_.*_in(\d+)_out(\d+).*\.csv$")

data_by_input_tokens: Dict[int, List[Tuple[int, float]]] = {k: [] for k in INPUT_TOKEN_LIST}

# print("Scanning files in:", dir_path)
found_any = False
for fname in sorted(os.listdir(dir_path)):
    match = pattern.search(fname)
    if match:
        in_tokens, _ = map(int, match.groups())
        if in_tokens in INPUT_TOKEN_LIST:
            found_any = True
            df = pd.read_csv(dir_path / fname)
            # Use output_tokens from file, not filename
            out_tokens = int(df['output_tokens'].iloc[0])
            val = df['decode_time'].iloc[0]
            latency = val / 1000.0 if val > 1000 else val
            # print(f"Found: {fname} | input_tokens={in_tokens} | output_tokens={out_tokens}")
            data_by_input_tokens[in_tokens].append((out_tokens, latency))

def combine_all_csvs(output_path: Path):
    """Combine all detailed_results_*.csv files in the directory into one CSV."""
    import glob
    all_files = glob.glob(str(dir_path / "detailed_results_*.csv"))
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"Combined CSV written to {output_path}")
    else:
        print("No CSV files found to combine.")

FIGSIZE = (6, 4)
LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 12
TITLE_FONT_SIZE = 12

plt.rcParams.update({'font.size': LABEL_FONT_SIZE})

if __name__ == "__main__":
    # Combine all detailed results into one CSV for inspection
    combine_all_csvs(Path(__file__).parent / "llama8b_decode_latency_combined.csv")
    if not found_any:
        print("No matching detailed_results_*.csv files found in directory!")
    else:
        # Combined plot for all input token contexts
        plt.figure(figsize=FIGSIZE)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:gray', 'tab:olive']
        for idx, in_tokens in enumerate(INPUT_TOKEN_LIST):
            data_points = data_by_input_tokens[in_tokens]
            if not data_points:
                continue
            data_points = sorted(data_points, key=lambda x: x[0])
            out_tokens_sorted = [x[0] for x in data_points]
            latencies_sorted = [x[1] for x in data_points]
            if in_tokens == 511:
                label_val = 512
            elif in_tokens == 1013:
                label_val = 1024
            else:
                label_val = in_tokens
            plt.plot(out_tokens_sorted, latencies_sorted, marker='o', linestyle='None', color=colors[idx], label=f'{label_val}')
            if len(out_tokens_sorted) > 1:
                coeffs = np.polyfit(out_tokens_sorted, latencies_sorted, 1)
                x_extrap = np.linspace(min(out_tokens_sorted), max(out_tokens_sorted)*1.5, 100)
                y_extrap = np.polyval(coeffs, x_extrap)
                plt.plot(x_extrap, y_extrap, linestyle='--', alpha=0.7, color=colors[idx])
        plt.xlabel("Output Tokens", fontsize=LABEL_FONT_SIZE)
        plt.ylabel("Decode Latency (s)", fontsize=LABEL_FONT_SIZE)
        plt.legend(title="Input Tokens", fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE, loc='lower right')
        plt.xticks(fontsize=TICK_FONT_SIZE)
        plt.yticks(fontsize=TICK_FONT_SIZE)
        plt.tight_layout()
        out_path = Path(__file__).parent / "llama8b_decode_latency_vs_output_all_inputs.pdf"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close()
        
        for idx, in_tokens in enumerate(INPUT_TOKEN_LIST):
            data_points = data_by_input_tokens[in_tokens]
            if not data_points:
                continue
            data_points = sorted(data_points, key=lambda x: x[0])
            out_tokens_sorted = [x[0] for x in data_points]
            latencies_sorted = [x[1] for x in data_points]
            plt.figure(figsize=FIGSIZE)
            plt.plot(out_tokens_sorted, latencies_sorted, marker='o', linestyle='None', color='tab:blue', label=f'Input {in_tokens}')
            if len(out_tokens_sorted) > 1:
                coeffs = np.polyfit(out_tokens_sorted, latencies_sorted, 1)
                x_extrap = np.linspace(min(out_tokens_sorted), max(out_tokens_sorted)*1.5, 100)
                y_extrap = np.polyval(coeffs, x_extrap)
                plt.plot(x_extrap, y_extrap, linestyle='--', alpha=0.7, color='tab:blue')
            plt.xlabel("Output Tokens", fontsize=LABEL_FONT_SIZE)
            plt.ylabel("Decode Latency (s)", fontsize=LABEL_FONT_SIZE)
            plt.legend(fontsize=LEGEND_FONT_SIZE)
            plt.xticks(fontsize=TICK_FONT_SIZE)
            plt.yticks(fontsize=TICK_FONT_SIZE)
            plt.tight_layout()
            out_path = Path(__file__).parent / f"llama8b_decode_latency_vs_output_in{in_tokens}.pdf"
            plt.savefig(out_path)
            print(f"Saved plot to {out_path}")
            plt.close()
