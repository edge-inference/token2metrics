"""
Extract input and output tokens from each model sheet in full_mmlu_by_model_tegra.xlsx (Jetson only) and save to input_token_list.xlsx.
Each sheet in the output will correspond to a model and contain both input_tokens and output_tokens columns.
"""
import pandas as pd
from pathlib import Path

def main():
    input_path = Path("../../data/mmlu/gpu/full_mmlu_by_model_tegra.xlsx")
    output_path = Path("../../data/mmlu/gpu/input_token_list.xlsx")
    xls = pd.ExcelFile(input_path)
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            if "input_tokens" in df.columns and "output_tokens" in df.columns:
                tokens_df = df[["input_tokens", "output_tokens"]].dropna().astype(int)
                tokens_df.to_excel(writer, sheet_name=sheet, index=False)
                print(f"Extracted {len(tokens_df)} input/output token pairs for {sheet}")
            else:
                print(f"Sheet {sheet} missing 'input_tokens' or 'output_tokens', skipping.")
        except Exception as e:
            print(f"Failed to process sheet {sheet}: {e}")
    writer.close()
    print(f"Saved input/output token lists to {output_path}")

if __name__ == "__main__":
    main()
