#!/bin/bash

DECODE_INPUT_DIR="../../../data/synthetic/gpu/decode"
DECODE_OUTPUT_DIR="../../../data/synthetic/gpu/decode/processed"

echo "Processing decode results from Tegra..."
echo "Input dir: $DECODE_INPUT_DIR"
echo "Output dir: $DECODE_OUTPUT_DIR"

python -m energy.cli --base-dir $DECODE_INPUT_DIR
python -m energy.cli --correlate --energy-dir $DECODE_INPUT_DIR --performance-file $DECODE_OUTPUT_DIR/all_results_by_model_*.xlsx
python -m energy.cli --insights --verbose
python -m energy.cli --fitting
