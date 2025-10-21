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
Scaling with synthetic data - Energy Analysis Runner - High-level interface.
"""
import sys
from pathlib import Path
from energy import EnergyAnalyzer


def main():
    """Run Figure 6 energy analysis with default paths."""
    print("🔋 Figure 6 Energy Analysis (MEAN-based)")
    print("=" * 40)
    
    # Default paths 
    energy_dir = "../tegra/scaling"
    performance_file = "results/tokens/scaling_summary.xlsx"
    output_dir = "outputs/energy"
    
    # Check if individual model files exist (we need these for MEAN data)
    tokens_dir = Path("results/tokens")
    model_files = list(tokens_dir.glob("DeepSeek-*.xlsx"))
    
    if not model_files:
        print(f"❌ Individual model Excel files not found in results/tokens/")
        print("Please run: python export.py first to generate model Excel files")
        return 1
    
    print(f"✅ Found {len(model_files)} model Excel files for MEAN-based correlation")
    
    # Run analysis
    analyzer = EnergyAnalyzer(output_dir)
    results = analyzer.run_analysis(energy_dir, performance_file)
    
    if not results:
        print("❌ Analysis failed")
        return 1
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📈 Generated {len(results['plots'])} visualizations")
    
    # Show key results
    df = results['correlations']
    insights = results['insights']
    
    print(f"\n📊 Summary:")
    print(f"  - {len(df)} data points correlated")
    print(f"  - {df['model_name'].nunique()} models analyzed")
    print(f"  - PS factors: {sorted(df['ps_factor'].unique())}")
    print(f"  - Data source: MEAN values across seeds")
    
    # Show that we're getting all seeds as MEAN
    print(f"\n🔍 Correlation Details:")
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        print(f"  {model}: {len(model_data)} configurations, seed={model_data['seed'].iloc[0]}")
    
    if 'model_efficiency' in insights:
        print(f"\n🏆 Most Energy Efficient Models:")
        efficiency_ranking = sorted(insights['model_efficiency'].items(), 
                                  key=lambda x: x[1]['energy_per_decode_j'])
        for i, (model, metrics) in enumerate(efficiency_ranking[:3], 1):
            print(f"  {i}. {model}: {metrics['energy_per_decode_j']:.3f} J/decode")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 