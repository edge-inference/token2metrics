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
Energy Insights and Visualization Module - Refactored.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from .analysis import PowerAnalyzer
from .visualization import PowerScalingCharts, EfficiencyHeatmap, DebugPlotter, IndividualPlotter
from .utils import PathManager, sort_models_by_size


class PowerInsightsAnalyzer:
    """Main analyzer for power consumption insights with modular visualization."""
    
    def __init__(self, target_token_ranges: List[int] = None, tolerance: int = 10, verbose: bool = False):
        """Initialize with target ranges and tolerance."""
        # Updated target ranges to match actual data from figure2 (1-token output tests)
        self.target_token_ranges = target_token_ranges or [3, 24, 128, 256, 384, 512, 640, 1013]
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Initialize core analyzer
        self.analyzer = PowerAnalyzer(self.target_token_ranges, self.tolerance, verbose)
        
        # Initialize visualization components
        self.power_charts = PowerScalingCharts()
        self.heatmap = EfficiencyHeatmap()
        self.debug_plotter = DebugPlotter()
        self.individual_plotter = IndividualPlotter()
    
    def run_complete_analysis(self, correlation_file: str = None) -> Dict[str, Any]:
        """Run complete power insights analysis with all visualizations."""
        # Auto-detect correlation file if not provided
        if not correlation_file:
            correlation_file = self._auto_detect_correlation_file()
            if not correlation_file:
                raise FileNotFoundError("No correlation file found. Please run correlation analysis first.")
        
        print(f"Auto-detected correlation file: {correlation_file}")
        
        # Load and process data
        load_counts = self.analyzer.load_correlation_data(correlation_file)
        filtered_df = self.analyzer.filter_questions_by_token_ranges()
        analysis_df = self.analyzer.generate_power_analysis()
        
        if analysis_df.empty:
            print("No data found for analysis")
            return {}
        
        # Generate all visualizations
        power_chart = self.power_charts.create(analysis_df)
        heatmap_chart = self.heatmap.create(filtered_df)
        individual_plots = self.individual_plotter.create_all(analysis_df)
        debug_chart = self.debug_plotter.create(self.analyzer.model_data)
        
        # Save insights data
        data_file = PathManager.get_output_path("power_insights.xlsx")
        self.analyzer.save_insights_data(data_file)
        
        # Print summary
        self._print_analysis_summary(analysis_df, load_counts)
        
        return {
            'power_chart': power_chart,
            'heatmap_chart': heatmap_chart,
            'individual_plots': individual_plots,
            'debug_chart': debug_chart,
            'data_file': data_file,
            'insights': self.analyzer.get_power_scaling_insights()
        }
    
    def _auto_detect_correlation_file(self) -> str:
        """Auto-detect the most recent correlation file."""
        results_dir = Path("energy_results")
        if not results_dir.exists():
            return ""
        
        correlation_files = list(results_dir.glob("energy_performance_correlation*.xlsx"))
        if not correlation_files:
            return ""
        
        # Return the most recent file
        return str(max(correlation_files, key=lambda f: f.stat().st_mtime))
    
    def _print_analysis_summary(self, analysis_df: pd.DataFrame, load_counts: Dict[str, int]):
        """Print analysis summary."""
        total_questions = sum(load_counts.values())
        print(f"\n=== POWER INSIGHTS ANALYSIS COMPLETE ===")
        print(f"Total questions analyzed: {len(analysis_df.groupby(['model_name', 'target_token_range']).size())}")
        print(f"Models: {', '.join(sort_models_by_size(load_counts.keys()))}")
        print(f"Token ranges: {self.target_token_ranges}")
        print(f"Charts and data saved in: insight_charts folder")


def run_insights_analysis(target_token_ranges: List[int] = None, 
                         correlation_file: str = None, 
                         verbose: bool = False) -> Dict[str, Any]:
    """
    Standalone function to run power insights analysis.
    
    Args:
        target_token_ranges: List of target token ranges to analyze
        correlation_file: Path to correlation file (auto-detected if None)
        verbose: Enable verbose output
    
    Returns:
        Dictionary with analysis results and file paths
    """
    if verbose:
        print("Starting power insights analysis...")
    
    analyzer = PowerInsightsAnalyzer(target_token_ranges)
    results = analyzer.run_complete_analysis(correlation_file)
    
    if verbose and results:
        insights = results.get('insights', {})
        if 'model_efficiency_rankings' in insights:
            print("\n=== MODEL EFFICIENCY RANKINGS ===")
            for i, model in enumerate(insights['model_efficiency_rankings'], 1):
                print(f"{i}. {model['model_name']}: {model['energy_per_token_mean']:.4f} J/token")
    
    return results 