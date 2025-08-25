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

# Find the performance file with glob expansion
PERF_FILE=$(ls "$PREFILL_OUTPUT_DIR"/all_results_by_model_*.xlsx 2>/dev/null | head -n 1)
if [ -n "$PERF_FILE" ]; then
    $PYTHON -m energy.cli --correlate --energy-dir "$PREFILL_INPUT_DIR" --performance-file "$PERF_FILE"
else
    echo "! Warning: No performance file found matching $PREFILL_OUTPUT_DIR/all_results_by_model_*.xlsx"
    echo "Skipping correlation analysis"
fi

$PYTHON -m energy.cli --insights --verbose

LATEST_CORR=$(ls -1 "$ROOT_OUTPUTS_DIR"/energy_performance_correlation*.xlsx 2>/dev/null | tail -n 1 || true)
if [ -n "$LATEST_CORR" ]; then
    $PYTHON -m energy.cli --fitting --correlation-file "$LATEST_CORR"
else
    echo "! Warning: No correlation file found for fitting analysis"
fi

# Generate lookup tables last (if script exists)
if [ -f "generate_lookup_table.py" ]; then
    $PYTHON generate_lookup_table.py
fi