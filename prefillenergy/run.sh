#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

REPO_ROOT=$(cd ../../.. && pwd)
PREFILL_INPUT_DIR="$REPO_ROOT/data/synthetic/gpu/prefill"
PREFILL_OUTPUT_DIR="$REPO_ROOT/data/synthetic/gpu/prefill/processed"
ROOT_OUTPUTS_DIR="$REPO_ROOT/outputs/prefill"
mkdir -p "$ROOT_OUTPUTS_DIR"

PYTHON=${PYTHON:-python3}

echo "Processing prefill results from Tegra..."
echo "Input dir: $PREFILL_INPUT_DIR"
echo "Output dir: $PREFILL_OUTPUT_DIR"

$PYTHON -m energy.cli --base-dir "$PREFILL_INPUT_DIR"
$PYTHON -m energy.cli --correlate --energy-dir "$PREFILL_INPUT_DIR" --performance-file "$PREFILL_OUTPUT_DIR/all_results_by_model_*.xlsx"

# Copy correlation Excel to root outputs with suffix
LATEST_CORR=$(ls -1 output/energy_performance_correlation*.xlsx 2>/dev/null | tail -n 1 || true)
if [ -n "$LATEST_CORR" ]; then
    cp "$LATEST_CORR" "$ROOT_OUTPUTS_DIR/energy_performance_correlation_prefill.xlsx"
fi
$PYTHON -m energy.cli --insights --verbose
$PYTHON -m energy.cli --fitting --correlation-file "$ROOT_OUTPUTS_DIR/energy_performance_correlation_prefill.xlsx"

# Generate lookup tables last (if script exists)
if [ -f "generate_lookup_table.py" ]; then
    $PYTHON generate_lookup_table.py
fi