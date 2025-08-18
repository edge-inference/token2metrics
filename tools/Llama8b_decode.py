"""
Batch predict decode latency for Llama-8B (DeepSeek-R1-Distill-Llama-8B) using selected input/output token pairs.
Each point will have a different marker/icon for each input token, and a second legend will show the mapping.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')
from src.modeling.pipeline import CompletePipelineTrainer

plt.rcParams.update({'font.size': 16})

MODEL_SIZE = "8B"
SHEET_NAME = "DeepSeek-R1-Distill-Llama-8B"
MARKERS = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', 'H', 'd', 'p']


def main():
    parser = argparse.ArgumentParser(description="Batch predict decode latency for Llama-8B with per-point markers.")
    parser.add_argument('--input', type=Path, default=Path('datasets/8Binput_token_list.xlsx'), help='Input token list Excel file')
    parser.add_argument('--output', type=Path, default=Path('outputs/8Bdecode.pdf'), help='Output plot file (PDF)')
    parser.add_argument('--model-path', type=Path, default=Path('outputs'), help='Path to saved predictors')
    args = parser.parse_args()

    xls = pd.ExcelFile(args.input)
    df = pd.read_excel(xls, sheet_name=SHEET_NAME)
    input_tokens = df['input_tokens'].dropna().astype(int).values
    output_tokens = df['output_tokens'].dropna().astype(int).values
    min_len = min(len(input_tokens), len(output_tokens))
    input_tokens = input_tokens[:min_len]
    output_tokens = output_tokens[:min_len]

    trainer = CompletePipelineTrainer(output_dir=args.model_path)
    predictor = trainer.get_predictor(MODEL_SIZE, "decode")
    preds = []
    for t_in, t_out in zip(input_tokens, output_tokens):
        result = predictor.predict_latency(int(t_in), int(t_out))
        preds.append(result['decode_latency_seconds'])

    plt.figure(figsize=(7, 5))
    handles = []
    for i, (t_in, t_out, pred) in enumerate(zip(input_tokens, output_tokens, preds)):
        marker = MARKERS[i % len(MARKERS)]
        h = plt.scatter([t_out], [pred], marker=marker, s=60, label=f"Input {t_in}")
        handles.append((h, t_in))
    plt.xlabel('Output Tokens (Decode)')
    plt.ylabel('Decode Latency (s)')
    # Main legend: all points
    plt.legend([h for h, _ in handles], [f"Input {t}" for _, t in handles], title="Input Tokens", loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved Llama-8B decode plot with per-point markers to {args.output}")

if __name__ == "__main__":
    main()
