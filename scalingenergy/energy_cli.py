#!/usr/bin/env python3
"""
Figure 6 Energy Analysis CLI - simple and focused interface.
"""
import argparse
import sys
from pathlib import Path
from energy import EnergyCollector, EnergyCorrelator, EnergyAnalyzer


def run_parse_energy(energy_dir: str, output_file: str = "energy_parsed.csv") -> None:
    """Parse energy CSV files and save results."""
    print("🔍 Parsing Energy Files")
    print("-" * 30)
    
    collector = EnergyCollector(energy_dir)
    df = collector.process_all_files()
    
    if df.empty:
        print("❌ No energy files found or parsed")
        return
    
    # Save results
    df.to_csv(output_file, index=False)
    
    print(f"✅ Parsed {len(df)} energy measurements")
    print(f"📊 Models: {df['model_name'].nunique()}")
    print(f"📁 Saved to: {output_file}")


def run_correlate_data(energy_dir: str, performance_file: str, 
                      output_file: str = "energy_correlations.xlsx") -> None:
    """Correlate energy and performance data."""
    print("🔗 Correlating Energy & Performance")
    print("-" * 35)
    
    correlator = EnergyCorrelator(energy_dir, performance_file)
    energy_loaded, perf_loaded = correlator.load_data()
    
    if not energy_loaded:
        print("❌ Failed to load energy data")
        return
    
    if not perf_loaded:
        print("❌ Failed to load performance data")
        return
    
    df = correlator.correlate_data()
    if df.empty:
        print("❌ No correlations found")
        return
    
    # Save results
    correlator.save_results(df, output_file)
    
    print(f"✅ Correlated {len(df)} data points")
    print(f"📊 Models: {df['model_name'].nunique()}")
    print(f"🔧 PS factors: {sorted(df['ps_factor'].unique())}")
    
    # Show matching statistics
    match_stats = df['match_method'].value_counts()
    print("\n📈 Matching Statistics:")
    for method, count in match_stats.items():
        print(f"  {method}: {count}")


def run_full_analysis(energy_dir: str, performance_file: str, 
                     output_dir: str = "outputs") -> None:
    """Run complete energy analysis with visualizations."""
    print("🔋 Complete Energy Analysis")
    print("-" * 30)
    
    analyzer = EnergyAnalyzer(output_dir)
    results = analyzer.run_analysis(energy_dir, performance_file)
    
    if not results:
        print("❌ Analysis failed")
        return
    
    print(f"📈 Generated {len(results['plots'])} visualizations")
    print(f"💡 Insights saved to {output_dir}/energy_insights.json")
    
    insights = results['insights']
    if 'model_efficiency' in insights:
        print("\n🏆 Top Energy Efficient Models:")
        efficiency_ranking = sorted(insights['model_efficiency'].items(), 
                                  key=lambda x: x[1]['energy_per_decode_j'])
        for i, (model, metrics) in enumerate(efficiency_ranking[:3], 1):
            print(f"  {i}. {model}: {metrics['energy_per_decode_j']:.3f} J/decode")


def main():
    parser = argparse.ArgumentParser(
        description="Figure 6 Energy Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Parse 
    parse_parser = subparsers.add_parser('parse', help='Parse energy CSV files')
    parse_parser.add_argument('--energy-dir', required=True,
                             help='Directory containing energy CSV files')
    parse_parser.add_argument('--output', default='energy_parsed.csv',
                             help='Output CSV file (default: energy_parsed.csv)')
    
    # Correlate 
    corr_parser = subparsers.add_parser('correlate', help='Correlate energy and performance data')
    corr_parser.add_argument('--energy-dir', required=True,
                            help='Directory containing energy CSV files')
    corr_parser.add_argument('--performance-file', required=True,
                            help='Performance results Excel file')
    corr_parser.add_argument('--output', default='energy_correlations.xlsx',
                            help='Output Excel file (default: energy_correlations.xlsx)')
    
    analyze_parser = subparsers.add_parser('analyze', help='Run complete energy analysis')
    analyze_parser.add_argument('--energy-dir', required=True,
                               help='Directory containing energy CSV files')
    analyze_parser.add_argument('--performance-file', required=True,
                               help='Performance results Excel file')
    analyze_parser.add_argument('--output-dir', default='outputs',
                               help='Output directory (default: outputs)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'parse':
            run_parse_energy(args.energy_dir, args.output)
        
        elif args.command == 'correlate':
            run_correlate_data(args.energy_dir, args.performance_file, args.output)
        
        elif args.command == 'analyze':
            run_full_analysis(args.energy_dir, args.performance_file, args.output_dir)
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 