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

# Natural-Plan Processor Setup Script
# Sets up the processor environment and runs initial analysis

set -e  # Exit on any error

echo "🚀 Natural-Plan Processor Setup"
echo "==============================="
echo ""

# Check if we're in the right directory
if [ ! -f "processor/analyze_results.py" ]; then
    echo "❌ Please run this script from the natural-plan project root directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing processor dependencies..."
pip install -r processor/requirements.txt

# Validate results directory
echo ""
echo "🔍 Validating results directory..."
python processor/validate_results.py

# Check for missing evaluations
echo ""
echo "📋 Checking for missing evaluations..."
python processor/check_coverage.py

# Ask if user wants to run batch analysis
echo ""
read -p "🚀 Run comprehensive batch analysis now? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Running batch analysis..."
    python processor/batch_analysis.py
    
    echo ""
    echo "✅ Setup and analysis complete!"
    echo "📁 Check processor/analysis_output/ for results"
    echo "📈 Plots saved in processor/analysis_output/plots/"
else
    echo "✅ Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  • Run: python processor/batch_analysis.py"
    echo "  • Or use: make batch-analysis"
    echo "  • Check missing evals: make check-missing"
fi

echo ""
echo "📚 See processor/README.md for full documentation"