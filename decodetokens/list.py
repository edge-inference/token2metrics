"""
List all Llama 8B decode result files and extract input/output token counts from filenames.
"""
import os
import re
from typing import List, Tuple

# Directory containing decode results
DATA_DIR = "/home/modfi/models/token2metrics/datasets/figure4"

# Corrected regex pattern for matching detailed results files
FILENAME_PATTERN = re.compile(r"detailed_results_.*_in(\d+)_out(\d+).*\.csv$")

def list_decode_files(data_dir: str) -> List[Tuple[str, int, int]]:
    files = os.listdir(data_dir)
    # print(f"All files in directory:")
    # for f in files:
    #     print(f"  {f}")
    print("\nMatching files:")
    matches = []
    for fname in files:
        match = FILENAME_PATTERN.search(fname)
        if match:
            in_tok = int(match.group(1))
            out_tok = int(match.group(2))
            print(f"[MATCH] {fname} (in={in_tok}, out={out_tok})")
            matches.append((fname, in_tok, out_tok))
        # else:
            # print(f"[NO MATCH] {fname}")
    return matches

def main() -> None:
    print(f"Listing all Llama 8B decode result files in: {DATA_DIR}")
    list_decode_files(DATA_DIR)

if __name__ == "__main__":
    main()
