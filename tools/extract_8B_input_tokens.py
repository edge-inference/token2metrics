"""
Extract input/output token pairs from input_token_list.xlsx for the 8B model (DeepSeek-R1-Distill-Llama-8B),
selecting rows where input_tokens are closest to [1, 128, 256, 512].
Save result to 8Binput_token_list.xlsx for targeted prediction experiments.
"""
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    input_path = Path("datasets/input_token_list.xlsx")
    output_path = Path("datasets/8Binput_token_list.xlsx")
    sheet = "DeepSeek-R1-Distill-Llama-8B"
    target_inputs = [1, 128, 256, 512]

    df = pd.read_excel(input_path, sheet_name=sheet)
    input_tokens = df['input_tokens'].values
    output_tokens = df['output_tokens'].values
    selected_rows = []
    used_indices = set()
    for t in target_inputs:
        # Find the index of the closest input_token not already used
        idx = (np.abs(input_tokens - t)).argmin()
        # Avoid duplicates if two targets are close
        while idx in used_indices:
            input_tokens[idx] = np.inf  # Exclude this index
            idx = (np.abs(input_tokens - t)).argmin()
        used_indices.add(idx)
        selected_rows.append((input_tokens[idx], output_tokens[idx]))
    # Create DataFrame and save
    out_df = pd.DataFrame(selected_rows, columns=["input_tokens", "output_tokens"])
    # Save with correct sheet name
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Saved selected input/output token pairs to {output_path} with sheet '{sheet}'")

if __name__ == "__main__":
    main()
