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
Energy Analysis CLI

Command-line interface for energy consumption analysis and correlation.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .aggregate import aggregate_energy_metrics, generate_detailed_model_summary
from .correlate import EnergyPerformanceCorrelator
from .insights import PowerInsightsAnalyzer
from .utils import PathManager

try:
    from .fitting.model_fits import ModelFitter
    _HAS_MODEL_FITTER = True
except ImportError:
    _HAS_MODEL_FITTER = False

def run_energy_analysis(base_dir: str, verbose: bool = False, prefer_gpu: bool = True) -> None:
    """Run basic energy analysis."""
    print("Energy Analysis Pipeline")
    print("="*50)
    
    if verbose:
        print(f"Base directory: {base_dir}")
        print(f"Power metric preference: {'GPU' if prefer_gpu else 'CPU'}")
    
    # Generate detailed model summary with raw data
    model_summary = generate_detailed_model_summary(base_dir, prefer_gpu=prefer_gpu)
    
    if model_summary.empty:
        print("✗ No energy data found to analyze")
        return
    
    print("\nModel Summary:")
    print(model_summary.to_string(index=False))
    
    print(f"\n✓ Energy analysis complete!")
    output_dir = PathManager.get_output_dir()
    print(f"Results saved to {output_dir}")


def run_correlation_analysis(energy_dir: str, performance_file: str, verbose: bool = False) -> None:
    """Run energy-performance correlation analysis."""
    print("Energy-Performance Correlation Analysis")
    print("="*60)
    
    if verbose:
        print(f"Energy directory: {energy_dir}")
        print(f"Performance file: {performance_file}")
    
    # Check if files exist
    if not os.path.exists(energy_dir):
        print(f"✗ Error: Energy directory not found: {energy_dir}")
        return
    
    if not os.path.exists(performance_file):
        print(f"✗ Error: Performance file not found: {performance_file}")
        return
    
    # Create correlator and run analysis
    correlator = EnergyPerformanceCorrelator(energy_dir, performance_file)
    
    # Generate correlation analysis
    combined_df = correlator.generate_correlation_analysis()
    
    if combined_df.empty:
        print("✗ No correlation data generated")
        return
    
    # Generate summary statistics
    summary_df = correlator.generate_summary_statistics(combined_df)
    
    # Save results
    output_path = correlator.save_correlation_results(combined_df, summary_df)
    
    print("\n✓ Energy-Performance Correlation Complete!")
    print("="*60)
    print(f"Generated comprehensive analysis with {len(combined_df)} question-level correlations")
    print(f"Results saved to: {output_path}")
    print("\nThe Excel file contains:")
    print("  • Model Summary: Overall efficiency rankings")
    print("  • Individual Model Sheets: Question-level data")
    print("  • Subject Analysis: Performance by subject")


def run_insights_analysis(correlation_file: str = None, verbose: bool = False) -> None:
    """
    Run power insights analysis and visualization.
    
    Args:
        correlation_file: Path to correlation results Excel file (auto-detected if None)
        verbose: Enable verbose output
    """
    print("Power Insights Analysis")
    print("="*40)
    
    if not correlation_file:
        output_dir = PathManager.get_output_dir()
        if not output_dir.exists():
            print("✗ Error: output directory not found")
            print("Please run correlation analysis first: python -m energy.cli --correlate")
            return
        
        correlation_files = list(output_dir.glob('energy_performance_correlation*.xlsx'))
        if not correlation_files:
            print("✗ Error: No correlation files found")
            print("Please run correlation analysis first: python -m energy.cli --correlate")
            return
        
        correlation_file = str(sorted(correlation_files)[-1])
        print(f"Auto-detected correlation file: {correlation_file}")
    
    if verbose:
        print(f"Correlation file: {correlation_file}")
    
    analyzer = PowerInsightsAnalyzer(verbose=verbose)
    results = analyzer.run_complete_analysis(correlation_file)
    
    if results:
        print(f"\n✓ Generated insights for {len(results.get('insights', {}).get('model_efficiency_rankings', []))} models")
        print("Check insight_charts/ folder for visualizations")


def run_fitting_analysis(correlation_file: str = None, output_dir: str = "output/fitting", 
                        plot_individual: bool = False, verbose: bool = False, 
                        target_input_length: int = None) -> None:
    """Run energy and power fitting analysis using correlation file."""
    print("🔧 Energy and Power Fitting Analysis")
    print("="*50)
    
    if not correlation_file:
        possible_dirs = [Path('./output'), Path('./energy_results'), Path('./figure3/energy_results'), Path('./results')]
        energy_results_dir = None
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                energy_results_dir = dir_path
                break
        
        if not energy_results_dir:
            print("✗ Error: No energy results directory found")
            print("Please run correlation analysis first: python -m energy.cli --correlate")
            return
        
        correlation_files = list(energy_results_dir.glob('energy_performance_correlation*.xlsx'))
        if not correlation_files:
            print("✗ Error: No correlation files found")
            print("Please run correlation analysis first: python -m energy.cli --correlate")
            return
        
        correlation_file = str(sorted(correlation_files)[-1])
        print(f"Auto-detected input file: {correlation_file}")
    
    if verbose:
        print(f"Correlation file: {correlation_file}")
        print(f"Output directory: {output_dir}")
        if target_input_length:
            print(f"Target input length: {target_input_length} tokens")
    
    actual_input_length = None
    if target_input_length:
        try:
            import pandas as pd
            df_sample = pd.read_excel(correlation_file, sheet_name=1)  
            available_inputs = sorted(df_sample['input_tokens'].unique())
            actual_input_length = min(available_inputs, key=lambda x: abs(x - target_input_length))
            print(f"Using closest input length: {actual_input_length} tokens (target: {target_input_length})")
        except Exception as e:
            print(f" Warning: Could not determine input lengths: {e}, ignoring input filter")
    
    try:
        if not _HAS_MODEL_FITTER:
            print("✗ ModelFitter not available")
            return
            
        fitter = ModelFitter(output_dir)
        
        # Fit all models using correlation file
        print("Fitting models...")
        results = fitter.fit_all_models(correlation_file, input_length_filter=actual_input_length)
        
        if not results:
            print("✗ No models were successfully fitted!")
            return
        
        print(f"\n✓ Successfully fitted {len(results)} models:")
        for model_name, model_results in results.items():
            power_r2 = model_results.get('power_fit', {}).get('r2_score', 0.0)
            energy_r2 = model_results.get('energy_fit', {}).get('r2_score', 0.0) if model_results.get('energy_fit') else 0.0
            print(f"  {model_name} ({model_results['model_size']}): Power R² = {power_r2:.4f}, Energy R² = {energy_r2:.4f}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if plot_individual:
            print("Generating individual plots...")
            fitter.plot_individual_fits(results, str(output_path))
        
        print("Generating comparison plots...")
        fitter.generate_comparison_plots(results, str(output_path))
        
        print("Saving fitting summary...")
        summary_file = output_path / "fitting_summary.json"
        fitter.save_fitting_summary(results, str(summary_file))
        
        print("\n✓ Fitting Analysis Complete!")
        print("="*50)
        print(f"Results saved to: {output_path}")
        print(f"Summary file: {summary_file}")
        print(f"📈 Comparison plots: {output_path / 'power_consumption_fits.pdf'} & {output_path / 'energy_j_token_fits.pdf'}")
        
    except Exception as e:
        print(f"✗ Error during fitting analysis: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Energy Analysis and Correlation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic energy analysis for CPU decode data (default)
  python -m energy.cli --base-dir ./cpu_decode
  
  # Basic energy analysis with GPU preference
  python -m energy.cli --base-dir ./cpu_decode --prefer-gpu
  
  # Energy-performance correlation with CPU decode data
  python -m energy.cli --correlate --energy-dir ./cpu_decode 
  
  # Energy-performance correlation with GPU preference
  python -m energy.cli --correlate --energy-dir ./cpu_decode --prefer-gpu
  
  # Power insights and visualizations
  python -m energy.cli --insights
  
  # Energy and power fitting analysis
  python -m energy.cli --fitting
  
  # Fitting with individual model plots
  python -m energy.cli --fitting --plot-individual
  
  # Verbose output with GPU preference
  python -m energy.cli --correlate --verbose --prefer-gpu
        """
    )
    
    parser.add_argument(
        '--base-dir',
        default='./cpu_decode',
        help='Base directory containing energy CSV files (default: ./cpu_decode)'
    )
    
    parser.add_argument(
        '--correlate',
        action='store_true',
        help='Run energy-performance correlation analysis'
    )
    
    parser.add_argument(
        '--insights',
        action='store_true',
        help='Generate power consumption insights and visualizations'
    )
    
    parser.add_argument(
        '--fitting',
        action='store_true',
        help='Run energy and power fitting analysis to find mathematical functions'
    )
    
    parser.add_argument(
        '--plot-individual',
        action='store_true',
        help='Generate individual fit plots for each model (use with --fitting)'
    )
    
    parser.add_argument(
        '--energy-dir',
        default='./cpu_decode',
        help='Directory containing energy CSV files for correlation (default: ./cpu_decode)'
    )
    
    parser.add_argument(
        '--performance-file',
        help='Path to performance results Excel file'
    )
    
    parser.add_argument(
        '--correlation-file',
        help='Path to correlation results Excel file for insights'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--prefer-gpu',
        action='store_true',
        default=True,
        help='Prioritize GPU power metrics over CPU (default for decode operations)'
    )
    
    parser.add_argument(
        '--prefer-cpu',
        action='store_true',
        help='Prioritize CPU power metrics over GPU (default: False for decode)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=int,
        help='Target input length for fitting (finds closest match in data, e.g., --input 512)'
    )
    
    args = parser.parse_args()
    
    try:
        
        if args.insights:
            run_insights_analysis(args.correlation_file, args.verbose)
        elif args.fitting:
            output_dir = PathManager.get_output_dir() / 'fitting'
            run_fitting_analysis(
                correlation_file=args.correlation_file,  
                output_dir=str(output_dir),
                plot_individual=args.plot_individual,
                verbose=args.verbose,
                target_input_length=args.input
            )
        elif args.correlate:
            if not args.performance_file:
                processed_results_dir = Path('../datasets/synthetic/gpu/decode/fine/processed_results')
                if processed_results_dir.exists():
                    excel_files = list(processed_results_dir.glob('all_results_by_model*.xlsx'))
                    if excel_files:
                        args.performance_file = str(sorted(excel_files)[-1])
                        print(f"Auto-detected performance file: {args.performance_file}")
                    else:
                        print("✗ Error: No performance file specified and none found in ../datasets/synthetic/gpu/decode/fine/processed_results/")
                        print("Use --performance-file to specify the path to your performance results Excel file")
                        sys.exit(1)
                else:
                    print("✗ Error: No performance file specified and ../datasets/synthetic/gpu/decode/fine/processed_results/ directory not found")
                    print("Use --performance-file to specify the path to your performance results Excel file")
                    sys.exit(1)
            
            run_correlation_analysis(args.energy_dir, args.performance_file, args.verbose)
        else:
            prefer_gpu = args.prefer_gpu and not args.prefer_cpu
            run_energy_analysis(args.base_dir, args.verbose, prefer_gpu=prefer_gpu)
            
    except KeyboardInterrupt:
        print("\n! Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
