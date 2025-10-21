#!/bin/bash

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

PERF_FILE=$(ls "$DECODE_OUTPUT_DIR"/all_results_by_model_*.xlsx 2>/dev/null | head -n 1)
if [ -n "$PERF_FILE" ]; then
    $PYTHON -m energy.cli --correlate --energy-dir "$DECODE_INPUT_DIR" --performance-file "$PERF_FILE"
else
    echo "! Warning: No performance file found matching $DECODE_OUTPUT_DIR/all_results_by_model_*.xlsx"
    echo "Skipping correlation analysis"
fi

$PYTHON -m energy.cli --insights --verbose

LATEST_CORR=$(ls -1 "$ROOT_OUTPUTS_DIR"/energy_performance_correlation*.xlsx 2>/dev/null | tail -n 1 || true)
if [ -n "$LATEST_CORR" ]; then
    $PYTHON -m energy.cli --fitting --correlation-file "$LATEST_CORR"
else
    echo "! Warning: No correlation file found for fitting analysis"
fi

if [ -f "empirical_data.py" ]; then
    $PYTHON empirical_data.py
fi
if [ -f "generate_lookup_table.py" ]; then
    $PYTHON generate_lookup_table.py
fi
