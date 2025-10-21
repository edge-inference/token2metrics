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
Plot output tokens vs. decode latency for all models (server side only).
Each model gets its own figure.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from configs.qwen_1_5b import DATA_CONFIG as QWEN_1_5B_DATA
from configs.llama_8b import DATA_CONFIG as LLAMA_8B_DATA
from configs.qwen_14b import DATA_CONFIG as QWEN_14B_DATA

MODELS = [
    ("Qwen-1.5B", QWEN_1_5B_DATA, "DeepSeek-R1-Distill-Qwen-1_5B"),
    ("LLaMA-8B", LLAMA_8B_DATA, "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen-14B", QWEN_14B_DATA, "DeepSeek-R1-Distill-Qwen-14B"),
]


def plot_server_decode(model_name, data_config, server_sheet, save_dir):
    server_df = pd.read_excel(data_config.server_data_path, sheet_name=server_sheet)
    n_server = len(server_df)
    plt.figure(figsize=(7, 5))
    plt.scatter(server_df["output_tokens"], server_df["decode_time"] / 1000.0, alpha=0.4, label=f"Server (n={n_server})", s=10)
    plt.xlabel("Output Tokens (Decode)")
    plt.ylabel("Decode Latency (s)")
    plt.title(f"{model_name}: Output Tokens vs. Decode Latency (Server)")
    plt.legend()
    plt.tight_layout()
    out_path = Path(save_dir) / f"{model_name.replace(' ', '_').lower()}_server_decode.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved decode plot for {model_name} to {out_path}")


def main():
    save_dir = Path("outputs/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    for model_name, data_config, server_sheet in MODELS:
        plot_server_decode(model_name, data_config, server_sheet, save_dir)

if __name__ == "__main__":
    main()
