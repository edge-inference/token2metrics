#!/usr/bin/env python3
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