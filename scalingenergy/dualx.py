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
Standalone Dual-Axis Plot Generator
Generates only the dual-axis visualization from existing correlation data.
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from plot_energy_results import EnergyPlotter

def main():
    parser = argparse.ArgumentParser(
        description='Generate dual-axis plot: Parallel Scaling Factor vs Power & GPU Utilization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python dualx.py --input results/energy/energy_correlations.xlsx --output plots/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='results/energy/energy_correlations.xlsx',
        help='Input correlation Excel file (default: results/energy/energy_correlations.xlsx)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='plots',
        help='Output directory for plot (default: plots)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        print("Please run the energy analysis first to generate the correlation file.")
        print("Or use: python test_3d_plot.py to test with sample data")
        return 1
    
    print("🔋 Dual-Axis Energy Plot Generator")
    print("=" * 35)
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    
    # Create plotter and load data
    plotter = EnergyPlotter(args.output)
    df = plotter.load_correlation_data(args.input)
    
    if df.empty:
        print("❌ No correlation data found")
        return 1
    
    # Check if GPU utilization data is available
    if 'avg_gpu_usage_pct' not in df.columns:
        print("❌ GPU utilization data not found in correlation file")
        print("Please ensure your energy CSV files contain GPU utilization metrics")
        return 1
    
    gpu_data_count = df['avg_gpu_usage_pct'].notna().sum()
    if gpu_data_count == 0:
        print("❌ No valid GPU utilization data found")
        print("All GPU utilization values are null/missing")
        return 1
    
    print(f"✅ Found {gpu_data_count}/{len(df)} records with GPU utilization data")
    
    # Generate dual-axis plot
    try:
        plot_path = plotter.plot_dual_axis_ps_power_gpu(df)
        
        print(f"\n✅ Dual-axis plot generated successfully!")
        print(f"📊 Plot saved to: {plot_path}")
        
        # Show summary of data plotted
        models_plotted = df['model_name'].unique()
        ps_factors = sorted(df['ps_factor'].unique())
        
        print(f"\n📈 Plot Summary:")
        print(f"   Models: {len(models_plotted)} ({', '.join(models_plotted)})")
        print(f"   PS Factors: {ps_factors}")
        print(f"   Power range: {df['avg_power_w'].min():.1f} - {df['avg_power_w'].max():.1f} W")
        print(f"   GPU utilization: {df['avg_gpu_usage_pct'].min():.1f} - {df['avg_gpu_usage_pct'].max():.1f} %")
        
        print(f"\n🎯 Dual-Axis Plot shows:")
        print(f"   - X-axis: Parallel Scaling Factor (logarithmic scale)")
        print(f"   - Left Y-axis: Power Consumption (W) - solid lines with circles")
        print(f"   - Right Y-axis: GPU Utilization (%) - dashed lines with squares")
        print(f"   - Colors represent different models")
        print(f"   - Clean, easy-to-read dual-axis format")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating dual-axis plot: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 