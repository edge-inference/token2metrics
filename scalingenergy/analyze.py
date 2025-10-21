#!/usr/bin/env python3

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

"""
Figure 6 Token Analysis Runner

Main runner for token-related analysis (decode latency vs PS scaling).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from analyzers.token_analyzer import TokenAnalyzer

def main():
    """Run Figure 6 token analysis."""
    print("🚀 Figure 6 Token Analysis")
    print("=" * 40)
    
    analyzer = TokenAnalyzer(excel_dir="results/tokens")
    
    analysis_results = analyzer.analyze_token_scaling()
    
    if not analysis_results:
        print("❌ No data found for analysis")
        return 1
    
    analyzer.create_scaling_plots(analysis_results, output_dir="outputs")
    
    analyzer.save_analysis_summary(analysis_results, output_dir="outputs")
    
    # Print summary
    analyzer.print_scaling_summary(analysis_results)
    
    print(f"\n✅ Token analysis complete!")
    print(f"📁 Results saved to: outputs/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 