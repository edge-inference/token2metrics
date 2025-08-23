#!/bin/bash

PREFILL_INPUT_DIR="../../../data/synthetic/gpu/prefill"
PREFILL_OUTPUT_DIR="../../../data/synthetic/gpu/prefill/processed"

echo "Processing prefill results from Tegra..."
echo "Input dir: $PREFILL_INPUT_DIR"
echo "Output dir: $PREFILL_OUTPUT_DIR"

python -m energy.cli --base-dir $PREFILL_INPUT_DIR
python -m energy.cli --correlate --energy-dir $PREFILL_INPUT_DIR --performance-file $PREFILL_OUTPUT_DIR/all_results_by_model_*.xlsx
python -m energy.cli --insights --verbose
python -m energy.cli --fitting