# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
