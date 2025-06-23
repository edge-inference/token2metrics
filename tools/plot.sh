#!/bin/bash
# Plot token-latency relationships for all models (server & Jetson)
set -e

cd "$(dirname "$0")/.."

python3 tools/plot_token_latency.py
