#!/bin/env python3
"""
Figure2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

script_dir = Path(__file__).parent.parent.resolve()
xlsx_candidates = list((script_dir / "datasets/synthetic/gpu/prefill_padded/processed_results").glob("all_results_by_model*.xlsx"))
if not xlsx_candidates:
    raise FileNotFoundError(
                f"No matching XLSX file found in processed_results/ (searched in {script_dir})"
        )
JETSON_XLSX_PATH = xlsx_candidates[0]

MODELS = [
    ("Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1_5B"),
    ("LLaMA-8B",  "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen-14B",  "DeepSeek-R1-Distill-Qwen-14B"),
]

MODEL_RELEASE_NAMES = {
    "Qwen-1.5B":"DSR1-Qwen-1.5B",
    "LLaMA-8B": "DSR1-LLaMA-8B",
    "Qwen-14B": "DSR1-Qwen-14B"
}

FIGSIZE = (7, 5)
LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 15
TICK_FONT_SIZE = 16
SCATTER_SIZE = 18
PLOT_TARGETS = [64, 128, 256, 384, 512, 1024, 1536, 2048, 3072, 4096]
plt.rcParams.update({'font.size': LABEL_FONT_SIZE})


def is_near_multiple_of_64(x, tol=2):
    return np.any(np.abs(x - np.round(x / 64) * 64) <= tol)


def filter_near_64(tokens, prefill, tol=2):
    mask = np.array([is_near_multiple_of_64(t, tol) for t in tokens])
    return tokens[mask], prefill[mask]


def fit_and_plot_quadratic(ax, tokens, prefill, label, color):
    coeffs = np.polyfit(tokens, prefill, 2)
    fit_fn = np.poly1d(coeffs)
    x_fit = np.linspace(tokens.min(), tokens.max(), 200)
    y_fit = fit_fn(x_fit)
    ax.plot(x_fit, y_fit, '--', color=color, linewidth=2, label=f"{label} Quadratic Fit")
    return coeffs


def select_closest_points(tokens, prefill, targets):
    selected_tokens = []
    selected_prefill = []
    for t in targets:
        idx = np.argmin(np.abs(tokens - t))
        selected_tokens.append(tokens[idx])
        selected_prefill.append(prefill[idx])
    return np.array(selected_tokens), np.array(selected_prefill)


def plot_combined(coeffs_dict, xls_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {
        "Qwen-1.5B": "#1f77b4",
        "LLaMA-8B": "#ff7f0e",
        "Qwen-14B": "#2ca02c"
    }
    
    fit_colors = {
        "Qwen-1.5B": "#d62728",
        "LLaMA-8B": "#9467bd",
        "Qwen-14B": "#8c564b"
    }
    xls = pd.ExcelFile(xls_path)

    for model_name, sheet_name in MODELS:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if 'ttft' not in df.columns:
            continue
        tokens = df["input_tokens"].values
        prefill = df["ttft"].values / 1000.0
        mask = tokens <= 4096
        tokens, prefill = tokens[mask], prefill[mask]

        label = MODEL_RELEASE_NAMES.get(model_name, model_name)
        ax.scatter(tokens, prefill, s=SCATTER_SIZE, alpha=0.7,
                   label=label, color=colors.get(model_name))

        tokens_all, prefill_all = select_closest_points(tokens, prefill, PLOT_TARGETS)
        
        tokens_fit, prefill_fit = tokens_all, prefill_all

        if model_name in coeffs_dict:
            coeffs = np.polyfit(tokens_fit, prefill_fit, 2)
            fit_fn = np.poly1d(coeffs)
            x_fit = np.linspace(tokens_fit.min(), tokens_fit.max(), 200)
            y_fit = fit_fn(x_fit)
            ax.plot(x_fit, y_fit, '--', color=colors.get(model_name), linewidth=2)

    ax.set_xlabel("Input Length", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Prefill Latency (s)", fontsize=LABEL_FONT_SIZE)
    ax.set_xlim([0, 1024])
    x_ticks = np.arange(0, 1025, 128)
    ax.set_xticks(x_ticks)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    fig.tight_layout()
    out_path = Path("plots") / "combined_prefill_fit_allpoints.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved fit plot to {out_path}")


def plot_all_data_detailed(xls_path):
    """Plot all data points without any cherry-picking or filtering for detailed analysis."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {
        "Qwen-1.5B": "#1f77b4",
        "LLaMA-8B": "#ff7f0e",
        "Qwen-14B": "#2ca02c"
    }
    
    xls = pd.ExcelFile(xls_path)

    for model_name, sheet_name in MODELS:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if 'ttft' not in df.columns:
            continue
        
        tokens = df["input_tokens"].values
        prefill = df["ttft"].values / 1000.0
        
        label = MODEL_RELEASE_NAMES.get(model_name, model_name)
        ax.scatter(tokens, prefill, s=SCATTER_SIZE, alpha=0.6,
                   label=label, color=colors.get(model_name))

    ax.set_xlabel("Input Length", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Prefill Latency (s)", fontsize=LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    out_path = Path("plots") / "all_data_detailed_plot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved detailed plot with all data points to {out_path}")


def main():
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True, parents=True)
    xls = pd.ExcelFile(JETSON_XLSX_PATH)
    coeffs_dict = {}

    for model_name, sheet_name in MODELS:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if 'ttft' not in df.columns:
            print(f"[ERROR] No 'ttft' column for {model_name}, skipping.")
            continue

        tokens = df["input_tokens"].values
        prefill = df["ttft"].values / 1000.0
        mask = tokens <= 4096
        tokens, prefill = tokens[mask], prefill[mask]

        tokens_all, prefill_all = select_closest_points(tokens, prefill, PLOT_TARGETS)

        tokens_fit, prefill_fit = tokens_all, prefill_all

        fig, ax = plt.subplots(figsize=FIGSIZE)
        label = MODEL_RELEASE_NAMES.get(model_name, model_name)
        
        ax.scatter(tokens, prefill, s=SCATTER_SIZE, alpha=0.5, label=label)

        if model_name in ("Qwen-1.5B", "LLaMA-8B", "Qwen-14B"):
            colors = {
                "Qwen-1.5B": "#1f77b4",
                "LLaMA-8B": "#ff7f0e",
                "Qwen-14B": "#2ca02c"
            }
            
            fit_color = colors.get(model_name, 'orange')
            
            coeffs = fit_and_plot_quadratic(ax, tokens_fit, prefill_fit, label, fit_color)
            coeffs_dict[model_name] = {"a": float(coeffs[0]),
                                      "b": float(coeffs[1]),
                                      "c": float(coeffs[2])}

        ax.set_xlabel("Input Length", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Prefill Latency (s)", fontsize=LABEL_FONT_SIZE)
        x_ticks = np.arange(0, 1025, 128)
        ax.set_xticks(x_ticks)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        fig.tight_layout()
        out_path = out_dir / f"{model_name.lower()}_prefill_fit_allpoints.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved fit plot for {model_name} to {out_path}")

    with open(out_dir / "fit_parameters_allpoints.json", "w") as f:
        json.dump(coeffs_dict, f, indent=2)
    print(f"Saved fit coefficients to {out_dir / 'fit_parameters_allpoints.json'}")

    plot_combined(coeffs_dict, JETSON_XLSX_PATH)
    plot_all_data_detailed(JETSON_XLSX_PATH)

if __name__ == "__main__":
    main()
