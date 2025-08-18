"""
Extract input/output token pairs from input_token_list.xlsx for the 8B model (DeepSeek-R1-Distill-Llama-8B),
selecting rows where input_tokens are closest to [1, 128, 256, 512].
Save result to 8Binput_token_list.xlsx for targeted prediction experiments.
"""
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    # input_path = Path("datasets/input_token_list.xlsx")
    input_path = Path("datasets/server/full_mmlu_by_model.xlsx")
    output_path = Path("datasets/14Binput_token_list.xlsx")
    sheet = "DeepSeek-R1-Distill-Qwen-14B"
    target_inputs = [128, 256, 512, 1024]

    df = pd.read_excel(input_path, sheet_name=sheet)
    # Ensure required columns exist
    required_cols = ["input_tokens", "question", "choices", "output_tokens", "question_id", "subject", "correct_answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input sheet.")
    selected_rows = []
    used_indices = set()
    for t in target_inputs:
        # Find all rows with input_tokens == t
        matches = df[df['input_tokens'] == t]
        if not matches.empty:
            for idx, row in matches.iterrows():
                if idx not in used_indices:
                    used_indices.add(idx)
                    selected_rows.append(row[required_cols].to_dict())
        else:
            # If no exact match, find closest not already used
            input_tokens = df['input_tokens'].values.copy()
            idx = (np.abs(input_tokens - t)).argmin()
            while idx in used_indices:
                input_tokens[idx] = np.inf
                idx = (np.abs(input_tokens - t)).argmin()
            used_indices.add(idx)
            row = df.iloc[idx]
            selected_rows.append(row[required_cols].to_dict())
    # Create DataFrame and save
    out_df = pd.DataFrame(selected_rows)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Saved selected input/output token pairs and metadata to {output_path} with sheet '{sheet}'")

if __name__ == "__main__":
    main()
