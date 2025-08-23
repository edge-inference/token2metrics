#!/bin/bash
set -e

cd "$(dirname "$0")/.."

rm -rf outputs/plots/*

python3 src/main.py
