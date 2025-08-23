#!/usr/bin/env python3
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