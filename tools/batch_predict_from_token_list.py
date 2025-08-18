"""
Batch predict decode latency for each model using input_tokens from input_token_list.xlsx.
Optionally, run for all models or a specific model. Plots all predictions on one plot.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from src.modeling.pipeline import CompletePipelineTrainer

plt.rcParams.update({'font.size': 16})

MODEL_NAME_TO_SIZE = {
    "DeepSeek-R1-Distill-Qwen-1_5B": "1.5B",
    "DeepSeek-R1-Distill-Llama-8B": "8B",
    "DeepSeek-R1-Distill-Qwen-14B": "14B",
    "L1-Qwen-1_5B-Max": "L1MAX"
}

# Mapping from model size to sheet name
MODEL_SIZE_TO_SHEET = {
    "1.5B": "DeepSeek-R1-Distill-Qwen-1_5B",
    "8B": "DeepSeek-R1-Distill-Llama-8B",
    "14B": "DeepSeek-R1-Distill-Qwen-14B",
    "L1MAX": "L1-Qwen-1_5B-Max"
}

PLOT_ORDER = [
    "L1-Qwen-1_5B-Max",
    "DeepSeek-R1-Distill-Qwen-1_5B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B"
]


def batch_predict(model_name, input_tokens, output_tokens, trainer):
    model_size = MODEL_NAME_TO_SIZE.get(model_name, model_name)  # fallback to model_name if not mapped
    predictor = trainer.get_predictor(model_size, "decode")
    preds = []
    for t_in, t_out in zip(input_tokens, output_tokens):
        result = predictor.predict_latency(int(t_in), int(t_out))
        preds.append(result['decode_latency_seconds'])
    return preds


def main():
    parser = argparse.ArgumentParser(description="Batch predict decode latency for models using input_token_list.xlsx")
    parser.add_argument('--input', type=Path, default=Path('datasets/input_token_list.xlsx'), help='Input token list Excel file')
    parser.add_argument('--output', type=Path, default=Path('outputs/batch_predictions.png'), help='Output plot file')
    parser.add_argument('--model-path', type=Path, default=Path('outputs'), help='Path to saved predictors')
    parser.add_argument('--model', type=str, help='Model name (sheet or size) to run, or omit for all')
    parser.add_argument('--output-tokens', type=int, default=1, help='Output tokens for decode prediction (default: 1)')
    args = parser.parse_args()

    xls = pd.ExcelFile(args.input)
    trainer = CompletePipelineTrainer(output_dir=args.model_path)
    if args.model:
        # Accept either model size or sheet name
        sheet = MODEL_SIZE_TO_SHEET.get(args.model, args.model)
        models = [sheet]
        model_keys = [args.model]
    else:
        # Sort models by preferred order, then any others
        all_sheets = xls.sheet_names
        models = [m for m in PLOT_ORDER if m in all_sheets] + [m for m in all_sheets if m not in PLOT_ORDER]
        model_keys = models
    plt.figure(figsize=(10, 6))
    for model, model_key in zip(models, model_keys):
        df = pd.read_excel(xls, sheet_name=model)
        input_tokens = df['input_tokens'].dropna().astype(int).values
        output_tokens = df['output_tokens'].dropna().astype(int).values
        # Use only rows where both are present
        min_len = min(len(input_tokens), len(output_tokens))
        input_tokens = input_tokens[:min_len]
        output_tokens = output_tokens[:min_len]
        preds = batch_predict(model, input_tokens, output_tokens, trainer)
        plt.scatter(output_tokens, preds, label=f"{model} (n={len(output_tokens)})", s=18, alpha=0.7)
    plt.xlabel('Output Tokens (Decode)')
    plt.ylabel('Decode Latency (s)')
    # No title
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf_path = args.output.with_suffix('.pdf')
    plt.savefig(pdf_path)
    print(f"Saved batch prediction plot to {pdf_path}")

if __name__ == "__main__":
    main()
