"""
Fit and plot quadratic models for prefill latency vs. input tokens for all Jetson models.
Saves fit coefficients and overlay plots in the prefill/ directory.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binned_statistic

JETSON_XLSX_PATH = Path("processed_results/all_results_by_model_20250624_133750.xlsx")

MODELS = [
    ("L1MAX",    "L1-Qwen-1_5B-Max"),
    ("Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1_5B"),
    ("LLaMA-8B",  "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen-14B",  "DeepSeek-R1-Distill-Qwen-14B"),
]

MODEL_RELEASE_NAMES = {
    "L1MAX":    "L1-Qwen-1.5B-Max",
    "Qwen-1.5B":"DSR1-Qwen-1.5B",
    "LLaMA-8B": "DSR1-LLaMA-8B",
    "Qwen-14B": "DSR1-Qwen-14B"
}

plt.rcParams.update({'font.size': 14})

RESULTS = {}


def fit_and_plot_quadratic(ax, tokens, prefill, label):
    coeffs = np.polyfit(tokens, prefill, 2)
    fit_fn = np.poly1d(coeffs)
    x_fit = np.linspace(tokens.min(), tokens.max(), 200)
    y_fit = fit_fn(x_fit)
    ax.plot(x_fit, y_fit, '--', color='orange', label=f"{label} Quadratic Fit")
    return coeffs

def fit_and_plot_stepwise(ax, tokens, prefill, label, n_bins=9):
    # Compute bin means using scipy's binned_statistic
    stat, bin_edges, _ = binned_statistic(tokens, prefill, statistic='mean', bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(stat)
    step_x = bin_centers[valid]
    step_y = stat[valid]
    
    if len(step_x) > 0:
        step_x_ext = np.concatenate(([tokens.min()], step_x, [tokens.max()]))
        step_y_ext = np.concatenate(([step_y[0]], step_y, [step_y[-1]]))
        ax.step(step_x_ext, step_y_ext, where='mid', label=f"{label} Stepwise Fit", linewidth=2, color='orange')
        return {"bin_centers": step_x_ext.tolist(), "mean_prefill": step_y_ext.tolist()}
    else:
        return {"bin_centers": [], "mean_prefill": []}


def plot_combined(coeffs_dict, xls):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "L1MAX": "tab:blue",
        "Qwen-1.5B": "tab:purple",
        "LLaMA-8B": "tab:green",
        "Qwen-14B": "tab:red"
    }
    for model_name, sheet_name in MODELS:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"[ERROR] Could not read sheet '{sheet_name}': {e}")
            continue
        if 'ttft' not in df.columns:
            continue
        tokens = df["input_tokens"].values
        prefill = df["ttft"].values / 1000.0
        # Cap at max input tokens of 650
        mask = tokens <= 650
        tokens = tokens[mask]
        prefill = prefill[mask]
        # Subsample 5-10 points per cluster using quantile binning
        n_clusters = 8
        df_sub = pd.DataFrame({'tokens': tokens, 'prefill': prefill})
        df_sub['bin'], bins = pd.qcut(df_sub['tokens'], q=n_clusters, retbins=True, labels=False, duplicates='drop')
        sampled_tokens = []
        sampled_prefill = []
        for i in range(n_clusters):
            in_bin = df_sub['bin'] == i
            bin_points = df_sub.loc[in_bin]
            if len(bin_points) > 0:
                n = min(10, max(5, len(bin_points)//2))
                sampled = bin_points.sample(n=n, random_state=42) if len(bin_points) > n else bin_points
                sampled_tokens.extend(sampled['tokens'].tolist())
                sampled_prefill.extend(sampled['prefill'].tolist())
        label = f"{MODEL_RELEASE_NAMES.get(model_name, model_name)}"
        ax.scatter(sampled_tokens, sampled_prefill, s=18, alpha=0.7, label=label, color=colors.get(model_name, None))
        # Add fit lines if present
        if "quadratic" in coeffs_dict[model_name]:
            coeffs = coeffs_dict[model_name]["quadratic"]
            fit_fn = np.poly1d([coeffs['a'], coeffs['b'], coeffs['c']])
            x_fit = np.linspace(min(sampled_tokens), max(sampled_tokens), 200)
            y_fit = fit_fn(x_fit)
            ax.plot(x_fit, y_fit, '--', color='gold', linewidth=2)
        if "stepwise" in coeffs_dict[model_name]:
            step = coeffs_dict[model_name]["stepwise"]
            ax.step(step["bin_centers"], step["mean_prefill"], where='mid', color='orange', linewidth=2)
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Prefill Latency (s)")
    ax.set_xlim([0, 650])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path("prefill") / "combined_prefill_fit.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved combined fit plot to {out_path}")


def plot_combined_1_5B(coeffs_dict, xls):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "L1MAX": "tab:blue",
        "Qwen-1.5B": "tab:purple"
    }
    for model_name, sheet_name in MODELS:
        if model_name not in ("L1MAX", "Qwen-1.5B"):
            continue
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"[ERROR] Could not read sheet '{sheet_name}': {e}")
            continue
        if 'ttft' not in df.columns:
            continue
        tokens = df["input_tokens"].values
        prefill = df["ttft"].values / 1000.0
        # Cap at max input tokens of 650
        mask = tokens <= 650
        tokens = tokens[mask]
        prefill = prefill[mask]
        # Subsample 5-10 points per cluster using quantile binning
        n_clusters = 8
        df_sub = pd.DataFrame({'tokens': tokens, 'prefill': prefill})
        df_sub['bin'], bins = pd.qcut(df_sub['tokens'], q=n_clusters, retbins=True, labels=False, duplicates='drop')
        sampled_tokens = []
        sampled_prefill = []
        for i in range(n_clusters):
            in_bin = df_sub['bin'] == i
            bin_points = df_sub.loc[in_bin]
            if len(bin_points) > 0:
                n = min(10, max(5, len(bin_points)//2))
                sampled = bin_points.sample(n=n, random_state=42) if len(bin_points) > n else bin_points
                sampled_tokens.extend(sampled['tokens'].tolist())
                sampled_prefill.extend(sampled['prefill'].tolist())
        label = f"{MODEL_RELEASE_NAMES.get(model_name, model_name)}"
        ax.scatter(sampled_tokens, sampled_prefill, s=18, alpha=0.7, label=label, color=colors.get(model_name, None))
        # Fit and plot linear for tokens <= 200, quadratic for > 200
        mask_linear = tokens <= 200
        mask_quad = tokens > 200
        if np.sum(mask_linear) > 2:
            lin_coeffs = np.polyfit(tokens[mask_linear], prefill[mask_linear], 1)
            lin_fn = np.poly1d(lin_coeffs)
            x_lin = np.linspace(tokens[mask_linear].min(), tokens[mask_linear].max(), 100)
            y_lin = lin_fn(x_lin)
            ax.plot(x_lin, y_lin, '-', color='gold', linewidth=2, label=f"{label} Linear Fit (≤200)")
        if np.sum(mask_quad) > 2:
            quad_coeffs = np.polyfit(tokens[mask_quad], prefill[mask_quad], 2)
            quad_fn = np.poly1d(quad_coeffs)
            x_quad = np.linspace(tokens[mask_quad].min(), tokens[mask_quad].max(), 100)
            y_quad = quad_fn(x_quad)
            ax.plot(x_quad, y_quad, '--', color='gold', linewidth=2, label=f"{label} Quadratic Fit (>200)")
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Prefill Latency (s)")
    ax.set_xlim([0, 650])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path("prefill") / "combined_1_5B_prefill_fit.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved combined 1.5B fit plot to {out_path}")


def main():
    out_dir = Path("prefill")
    out_dir.mkdir(exist_ok=True, parents=True)
    xls = pd.ExcelFile(JETSON_XLSX_PATH)
    coeffs_dict = {}
    for model_name, sheet_name in MODELS:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"[ERROR] Could not read sheet '{sheet_name}': {e}")
            continue
        if 'ttft' not in df.columns:
            print(f"[ERROR] No 'ttft' column found for {model_name}, skipping.")
            continue
        tokens = df["input_tokens"].values
        prefill = df["ttft"].values / 1000.0
        fig, ax = plt.subplots(figsize=(8, 6))
        label = f"{MODEL_RELEASE_NAMES.get(model_name, model_name)}"
        ax.scatter(tokens, prefill, s=16, alpha=0.5, label=label)
        coeffs_dict[model_name] = {}
        # Only plot quadratic for 1.5B models
        if model_name in ("L1MAX", "Qwen-1.5B"):
            coeffs = fit_and_plot_quadratic(ax, tokens, prefill, label)
            coeffs_dict[model_name]["quadratic"] = {
                "a": float(coeffs[0]),
                "b": float(coeffs[1]),
                "c": float(coeffs[2])
            }
        # Only plot stepwise for 8B and 14B
        if model_name in ("LLaMA-8B", "Qwen-14B"):
            stepwise = fit_and_plot_stepwise(ax, tokens, prefill, label)
            coeffs_dict[model_name]["stepwise"] = stepwise
        ax.set_xlabel("Input Tokens")
        ax.set_ylabel("Prefill Latency (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f"{model_name.lower()}_prefill_fit.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved fit plot for {model_name} to {out_path}")
    # Save coefficients to JSON
    import json
    with open(out_dir / "fit_parameters.json", "w") as f:
        json.dump(coeffs_dict, f, indent=2)
    print(f"Saved fit coefficients to {out_dir / 'fit_parameters.json'}")
    # Add combined plots
    plot_combined(coeffs_dict, xls)
    plot_combined_1_5B(coeffs_dict, xls)

if __name__ == "__main__":
    main()
