#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

REPO_ROOT=$(cd ../../.. && pwd)
DECODE_INPUT_DIR="$REPO_ROOT/data/synthetic/gpu/decode"
DECODE_OUTPUT_DIR="$REPO_ROOT/data/synthetic/gpu/decode/processed"
ROOT_OUTPUTS_DIR="$REPO_ROOT/outputs/decode"
mkdir -p "$ROOT_OUTPUTS_DIR"

PYTHON=${PYTHON:-python3}

echo "Processing decode results from Tegra..."
echo "Input dir: $DECODE_INPUT_DIR"
echo "Output dir: $DECODE_OUTPUT_DIR"

$PYTHON -m energy.cli --base-dir "$DECODE_INPUT_DIR"
$PYTHON -m energy.cli --correlate --energy-dir "$DECODE_INPUT_DIR" --performance-file "$DECODE_OUTPUT_DIR/all_results_by_model_*.xlsx"

# Copy correlation Excel to root outputs with suffix
LATEST_CORR=$(ls -1 output/energy_performance_correlation*.xlsx 2>/dev/null | tail -n 1 || true)
if [ -n "$LATEST_CORR" ]; then
    cp "$LATEST_CORR" "$ROOT_OUTPUTS_DIR/energy_performance_correlation_decode.xlsx"
fi
$PYTHON -m energy.cli --insights --verbose
$PYTHON -m energy.cli --fitting --correlation-file "$ROOT_OUTPUTS_DIR/energy_performance_correlation_decode.xlsx"

if [ -f "empirical_data.py" ]; then
    $PYTHON empirical_data.py
fi
if [ -f "generate_lookup_table.py" ]; then
    $PYTHON generate_lookup_table.py
fi
