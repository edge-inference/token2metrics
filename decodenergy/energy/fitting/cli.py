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
CLI for Energy and Power Fitting Analysis

Fits mathematical models to energy and power consumption data.
"""

import argparse
from pathlib import Path
from .model_fits import ModelFitter


def main():
    parser = argparse.ArgumentParser(description="Fit energy and power trends for different models")
    parser.add_argument(
        "--correlation-file", "-c",
        type=str,
        default="energy_results/energy_performance_correlation.xlsx",
        help="Path to energy performance correlation Excel file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="fitting",
        help="Output directory for fitting results"
    )
    parser.add_argument(
        "--plot-individual",
        action="store_true",
        help="Generate individual plots for each model"
    )
    parser.add_argument(
        "--plot-comparison",
        action="store_true",
        default=True,
        help="Generate comparison plots across all models (default: True)"
    )
    
    args = parser.parse_args()
    
    # Check if correlation file exists
    correlation_file = Path(args.correlation_file)
    if not correlation_file.exists():
        print(f"Error: Correlation file not found: {correlation_file}")
        print("Please run the energy processing pipeline first to generate the correlation file.")
        return 1
    
    print("🔧 Energy and Power Fitting Analysis")
    print("=" * 50)
    print(f"📊 Input file: {correlation_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print()
    
    # Initialize fitter
    fitter = ModelFitter()
    
    # Fit all models
    print("Fitting models...")
    results = fitter.fit_all_models(str(correlation_file))
    
    if not results:
        print("❌ No models were successfully fitted.")
        return 1
    
    # Print results summary
    print()
    print("✅ Successfully fitted models:")
    for model_name, model_results in results.items():
        power_r2 = model_results.get('power_fit', {}).get('r2_score', 0.0)
        energy_r2 = model_results.get('energy_fit', {}).get('r2_score', 0.0) if model_results.get('energy_fit') else 0.0
        print(f"  {model_name}: Power R² = {power_r2:.4f}, Energy (mJ) R² = {energy_r2:.4f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.plot_comparison:
        print()
        print("Generating comparison plots...")
        fitter.generate_comparison_plots(results, str(output_dir))
    
    if args.plot_individual:
        print("Generating individual plots...")
        fitter.plot_individual_fits(results, str(output_dir))
    
    # Save fitting summary
    print("Saving fitting summary...")
    summary_file = output_dir / "fitting_summary.json"
    fitter.save_fitting_summary(results, str(summary_file))
    
    print()
    print("🎉 Fitting Analysis Complete!")
    print("=" * 50)
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Summary file: {summary_file}")
    if args.plot_comparison:
        print(f"📈 Comparison plots: {output_dir / 'model_comparison_fits.png'}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 