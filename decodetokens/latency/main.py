"""
Plot decode latency vs. output tokens for all models.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json
import matplotlib.ticker as ticker
import os
DEFAULT_RESULTS_DIR = Path("../../datasets/synthetic/gpu/decode/fine/processed_results")

def find_latest_results_csv(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("all_results_consolidated_*.csv"))
    return files[-1] if files else None

RESULTS_CSV_PATH = find_latest_results_csv(DEFAULT_RESULTS_DIR)

if not RESULTS_CSV_PATH and os.environ.get("DECODE_RESULTS_CSV"):
    RESULTS_CSV_PATH = Path(os.environ["DECODE_RESULTS_CSV"])

FIGURE_WIDTH = 6
FIGURE_HEIGHT = 4
FONT_SIZE_LABELS = 14
FONT_SIZE_LEGEND = 10
FONT_SIZE_TICKS = 12

# Target input length for filtering (can be changed to generate plots for other input lengths)
TARGET_INPUT_LENGTH = 512
INPUT_LENGTH_TOLERANCE = 50

def linear_function(x, a, b):
    return a * x + b

def shorten_model_name(model_name):
    if 'Qwen-1_5B' in model_name:
        return 'DSR1-Qwen-1.5B'
    elif 'Qwen-14B' in model_name:
        return 'DSR1-Qwen-14B'
    elif 'Llama-8B' in model_name:
        return 'DSR1-Llama-8B'
    else:
        parts = model_name.replace('DeepSeek-R1-Distill-', '').split('-')
        if len(parts) >= 2:
            return f"DSR1-{parts[0]}-{parts[1]}"
        return f"DSR1-{model_name}"

def plot_all_models_decode_latency(save_dir):
    if not RESULTS_CSV_PATH or not RESULTS_CSV_PATH.exists():
        print(f"[ERROR] CSV file not found at {RESULTS_CSV_PATH}")
        return
    
    # Load consolidated CSV
    try:
        df_all = pd.read_csv(RESULTS_CSV_PATH)
        print(f"[INFO] Loaded {len(df_all)} data points from {RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not read CSV file: {e}")
        return
    
    # Filter to required columns
    required_cols = ['model_name', 'decode_time', 'output_tokens', 'input_tokens']
    if not all(col in df_all.columns for col in required_cols):
        print(f"[ERROR] Missing required columns. Found: {list(df_all.columns)}")
        return
    
    df_all = df_all[required_cols].dropna()
    
    # Filter for input tokens around TARGET_INPUT_LENGTH (looking for 522 or close values)
    input_filtered = df_all[
        (df_all['input_tokens'] >= TARGET_INPUT_LENGTH - INPUT_LENGTH_TOLERANCE) & 
        (df_all['input_tokens'] <= TARGET_INPUT_LENGTH + INPUT_LENGTH_TOLERANCE)
    ].copy()
    
    if len(input_filtered) == 0:
        print(f"[WARN] No data found for input length ~{TARGET_INPUT_LENGTH}. Available input lengths: {sorted(df_all['input_tokens'].unique())}")
        most_common_input = df_all['input_tokens'].mode().iloc[0]
        input_filtered = df_all[df_all['input_tokens'] == most_common_input].copy()
        print(f"[INFO] Using most common input length: {most_common_input}")
    
    models = input_filtered['model_name'].unique()
    print(f"[INFO] Found {len(models)} models: {list(models)}")
    print(f"[INFO] Using input length filter ~{TARGET_INPUT_LENGTH}: {len(input_filtered)} data points")
    print(f"[INFO] Input token range: {input_filtered['input_tokens'].min()}-{input_filtered['input_tokens'].max()}")
    print(f"[INFO] Output token range: {input_filtered['output_tokens'].min()}-{input_filtered['output_tokens'].max()}")
    
    df_all = input_filtered
    
    def get_model_sort_key(model_name):
        if '14B' in model_name:
            return 0
        elif '8B' in model_name:
            return 1
        elif '1.5B' in model_name:
            return 2
        else:
            return 999
    
    models = sorted(models, key=get_model_sort_key)
    
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    def get_model_color(model_name):
        if '14B' in model_name:
            return '#1f77b4'
        elif '8B' in model_name:
            return '#ff7f0e'
        elif '1.5B' in model_name:
            return '#2ca02c'
        else:
            return '#d62728'
    
    fit_results = {}
    
    for model_name in models:
        try:
            model_df = df_all[df_all['model_name'] == model_name].copy()
            
            decode_latency = model_df["decode_time"] / 1000.0
            output_tokens = model_df["output_tokens"]
            
            display_name = shorten_model_name(model_name)
            color = get_model_color(model_name)
            
            print(f"[INFO] Plotting {len(model_df)} points for {display_name}")
            
            ax.scatter(output_tokens, decode_latency, alpha=0.6, 
                      label=display_name, 
                      s=20, color=color)
            
            if len(output_tokens) > 1:
                try:
                    popt, _ = curve_fit(linear_function, output_tokens, decode_latency)
                    
                    x_min = 0
                    x_max = max(output_tokens.max(), 2048)
                    x_fit = np.linspace(x_min, x_max, 100)
                    y_fit = linear_function(x_fit, *popt)
                    
                    y_pred = linear_function(output_tokens, *popt)
                    r2 = 1 - np.sum((decode_latency - y_pred)**2) / np.sum((decode_latency - np.mean(decode_latency))**2)
                    
                    ax.plot(x_fit, y_fit, '--', linewidth=2, color=color)
                    
                    fit_results[display_name] = {
                        'slope': float(popt[0]),
                        'intercept': float(popt[1]),
                        'r2_score': float(r2),
                        'equation': f"y = {popt[0]:.6f}x + {popt[1]:.6f}",
                        'n_points': int(len(output_tokens))
                    }
                    
                except Exception as e:
                    print(f"[WARN] Could not fit decode data for {display_name}: {e}")
                    fit_results[display_name] = {
                        'slope': None,
                        'intercept': None,
                        'r2_score': None,
                        'equation': "Fit failed",
                        'n_points': int(len(output_tokens))
                    }
            
        except Exception as e:
            print(f"[ERROR] Failed to process {model_name}: {e}")
    
    ax.set_xlabel("Output Tokens", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Decode Latency (s)", fontsize=FONT_SIZE_LABELS)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
    
    # Use log scale for x-axis to handle wide range of token values
    ax.set_xscale('log')
    
    # Set axis limits based on all data
    all_output_tokens = df_all['output_tokens'].tolist()
    
    if all_output_tokens:
        x_min = min(all_output_tokens) * 0.8  # Add some padding
        x_max = max(all_output_tokens) * 1.2
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        
        
        ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=2, subs=None, numticks=20))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        
        # Rotate x-axis labels to prevent overlap
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Include input length in filename
    input_suffix = f"_input{TARGET_INPUT_LENGTH}" if TARGET_INPUT_LENGTH else ""
    plot_path = Path(save_dir) / f"all_models_decode_latency{input_suffix}.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"Saved combined plot to {plot_path}")
    
    json_path = Path(save_dir) / f"decode_latency_fit_results{input_suffix}.json"
    with open(json_path, 'w') as f:
        json.dump(fit_results, f, indent=2)
    print(f"Saved fit results to {json_path}")
    
    return fit_results

def main():
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fit_results = plot_all_models_decode_latency(save_dir)
    
    if fit_results:
        print(f"\n[INFO] Linear fit results:")
        for model, results in fit_results.items():
            if results['r2_score'] is not None:
                print(f"  {model}: {results['equation']}, R² = {results['r2_score']:.3f}")
            else:
                print(f"  {model}: Fit failed")
    
    print(f"\n[INFO] All outputs saved to {save_dir}")

if __name__ == "__main__":
    main()
