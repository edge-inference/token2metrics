#!/usr/bin/env python3
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