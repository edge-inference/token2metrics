#!/usr/bin/env python3
"""
Figure 6 Export Script

Single script to export scaling results to structured Excel files.
Replaces the redundant export_energy.py and export_all.py wrapper scripts.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from processors.core_parser import ScalingResultsProcessor
from exporters.excel_exporter import ExcelExporter
from energy.analyzer import EnergyAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Figure 6: Export scaling results to Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export everything (tokens + energy) - RECOMMENDED
  python export.py --all
  python export.py  # Same as --all (default)

  # Export only energy analysis
  python export.py --energy

  # Export only token results  
  python export.py --tokens

  # Custom input/output directories
  python export.py --all --input ../tegra/scaling --output ./custom_results
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Export both token results AND energy analysis (default)'
    )
    
    parser.add_argument(
        '--energy',
        action='store_true',
        help='Export only energy analysis Excel files'
    )
    
    parser.add_argument(
        '--tokens',
        action='store_true',
        help='Export only token results Excel files'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='../tegra/scaling',
        help='Input directory containing raw result files (default: ../tegra/scaling)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for Excel files (default: ./results)'
    )
    
    args = parser.parse_args()
    
    if not (args.energy or args.tokens):
        args.all = True
    
    print("📊 Figure 6 Export Script")
    print("=" * 50)
    
    print(f"🔍 Parsing results from: {args.input}")
    processor = ScalingResultsProcessor(args.input)
    results = processor.parse_all_results()
    
    if not results:
        print("❌ No results to export")
        sys.exit(1)
    
    print(f"✅ Loaded {len(results)} scaling results")
    
    if args.all:
        print("\n🚀 Exporting ALL results (tokens + energy)")
        export_all_results(results, args.output, args.input)
    elif args.energy:
        print("\n🔋 Exporting ENERGY analysis only")
        export_energy_only(results, args.output, args.input)
    elif args.tokens:
        print("\n🎯 Exporting TOKEN results only")
        export_tokens_only(results, args.output)
    
    print("\n✅ Export complete!")
    print(f"📁 Results saved in: {args.output}/")
    print("\n💡 Next step: python analyze.py")


def export_all_results(results, output_dir, input_dir):
    print("\n📊 STEP 1: Creating token results Excel files...")
    excel_exporter = ExcelExporter()
    tokens_dir = f"{output_dir}/tokens"
    excel_exporter.export_by_model_multisheet(results, tokens_dir)
    
    print("\n🔋 STEP 2: Creating energy analysis visualizations...")
    energy_analyzer = EnergyAnalyzer("outputs/energy")
    performance_file = f"{output_dir}/tokens/scaling_summary.xlsx"
    
    energy_results = energy_analyzer.run_analysis(input_dir, performance_file)
    
    # Print summaries
    print("\n📈 STEP 3: Analysis summaries...")
    print_overall_summary(results)
    
    if energy_results:
        print(f"\n🔋 Energy Analysis Results:")
        print(f"  📊 Correlated {len(energy_results['correlations'])} data points")
        print(f"  📈 Generated {len(energy_results['plots'])} visualizations")
        print(f"  🎯 Analyzed {energy_results['correlations']['model_name'].nunique()} models")
    
    print(f"\n📁 Token results: {tokens_dir}/")
    print(f"🔋 Energy analysis: outputs/energy/")


def export_energy_only(results, output_dir, input_dir):
    """Export only energy analysis ."""
    print("🔄 Creating token summary for energy correlation...")
    excel_exporter = ExcelExporter()
    tokens_dir = f"{output_dir}/tokens"
    excel_exporter.export_by_model_multisheet(results, tokens_dir)
    
    energy_analyzer = EnergyAnalyzer("outputs/energy")
    performance_file = f"{tokens_dir}/scaling_summary.xlsx"
    
    energy_results = energy_analyzer.run_analysis(input_dir, performance_file)
    
    if energy_results:
        print(f"\n🔋 Energy Analysis Results:")
        print(f"  📊 Correlated {len(energy_results['correlations'])} data points")
        print(f"  📈 Generated {len(energy_results['plots'])} visualizations")
        print(f"  🎯 Analyzed {energy_results['correlations']['model_name'].nunique()} models")
    
    print(f"🔋 Energy analysis: outputs/energy/")


def export_tokens_only(results, output_dir):
    """Export only token results."""
    excel_exporter = ExcelExporter()
    tokens_dir = f"{output_dir}/tokens"
    excel_exporter.export_by_model_multisheet(results, tokens_dir)
    print_overall_summary(results)
    print(f"📁 Token results: {tokens_dir}/")


def print_overall_summary(results):
    print("\n📊 OVERALL SCALING SUMMARY")
    print("-" * 40)
    
    models = {}
    for result in results:
        model_name = result.metadata.model_name
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(result)
    
    for model_name, model_results in models.items():
        accuracies = [r.metrics.accuracy for r in model_results]
        sample_counts = list(set(r.metadata.num_samples for r in model_results))
        
        print(f"\n{model_name}:")
        print(f"  📈 Accuracy: {min(accuracies):.1%} - {max(accuracies):.1%} (avg: {sum(accuracies)/len(accuracies):.1%})")
        print(f"  🔢 Sample counts: {sorted(sample_counts)}")
        print(f"  🔄 Total runs: {len(model_results)}")


if __name__ == "__main__":
    main() 