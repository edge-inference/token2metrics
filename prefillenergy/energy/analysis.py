"""
Core analysis module for energy consumption data processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
from .utils import PathManager


class PowerAnalyzer:
    """Core analyzer for power consumption patterns."""
    
    def __init__(self, target_token_ranges: List[int] = None, tolerance: int = 10):
        """Initialize analyzer with target ranges and tolerance."""
        self.target_token_ranges = target_token_ranges or [3, 24, 128, 256, 384, 512, 640, 1013]
        self.tolerance = tolerance
        self.model_data = {}
        self.filtered_data = pd.DataFrame()
        self.analysis_data = pd.DataFrame()
    
    def load_correlation_data(self, file_path: str) -> Dict[str, int]:
        """Load correlation data from Excel file."""
        print(f"Loading correlation data from: {file_path}")
        model_summary = pd.read_excel(file_path, sheet_name='Model_Summary')
        model_names = model_summary['model_name'].tolist()
        load_counts = {}
        for model_name in model_names:
            try:
                sheet_name = model_name.replace('-', '_')
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                self.model_data[model_name] = df
                load_counts[model_name] = len(df)
                print(f"  {model_name}: {len(df)} questions loaded")
            except Exception as e:
                print(f"  Warning: Could not load {model_name}: {e}")
        return load_counts
    
    def filter_questions_by_token_ranges(self) -> pd.DataFrame:
        """Use all available data points without filtering by token ranges."""
        print("Using all available data points (no token range filtering)")
        filtered_dfs = []
        for model_name, df in self.model_data.items():
            df_copy = df.copy()
            df_copy['target_token_range'] = df_copy['input_tokens']
            df_copy['model_name'] = model_name
            filtered_dfs.append(df_copy)
        self.filtered_data = pd.concat(filtered_dfs, ignore_index=True)
        print(f"Total questions found: {len(self.filtered_data)}")
        print(f"Input token ranges: {sorted(self.filtered_data['input_tokens'].unique())}")
        return self.filtered_data
    
    def generate_power_analysis(self) -> pd.DataFrame:
        """Generate power consumption analysis for all data points."""
        if self.filtered_data.empty:
            return pd.DataFrame()
        print("Generating power consumption analysis...")
        analysis_groups = self.filtered_data.groupby(['model_name', 'input_tokens'])
        analysis_results = []
        for (model, input_tokens), group in analysis_groups:
            tokens_per_second = group['tokens_per_second']
            result = {
                'model_name': model,
                'target_token_range': input_tokens,
                'input_tokens': input_tokens,
                'question_count': len(group),
                'avg_power_w_mean': group['avg_power_w'].mean(),
                'avg_power_w_std': group['avg_power_w'].std() if len(group) > 1 else 0,
                'energy_per_token_mean': group['energy_per_token'].mean(),
                'energy_per_token_std': group['energy_per_token'].std() if len(group) > 1 else 0,
                'tokens_per_second_mean': tokens_per_second.mean(),
                'tokens_per_second_std': tokens_per_second.std() if len(group) > 1 else 0,
                'total_energy_j_sum': group['total_energy_j'].sum(),
                'total_time_ms_sum': group['total_time_ms'].sum()
            }
            analysis_results.append(result)
        self.analysis_data = pd.DataFrame(analysis_results)
        print(f"Generated analysis for {len(self.analysis_data)} data points")
        return self.analysis_data
    
    def get_model_efficiency_rankings(self) -> pd.DataFrame:
        """Get model efficiency rankings based on energy per token."""
        if self.analysis_data.empty:
            return pd.DataFrame()
        model_efficiency = self.analysis_data.groupby('model_name').agg({
            'energy_per_token_mean': 'mean',
            'avg_power_w_mean': 'mean',
            'tokens_per_second_mean': 'mean',
            'question_count': 'sum'
        }).reset_index()
        model_efficiency = model_efficiency.sort_values('energy_per_token_mean')
        model_efficiency['efficiency_rank'] = range(1, len(model_efficiency) + 1)
        return model_efficiency
    
    def get_power_scaling_insights(self) -> Dict[str, Any]:
        """Generate insights about power scaling patterns."""
        if self.analysis_data.empty:
            return {}
        insights = {}
        power_by_range = self.analysis_data.groupby('target_token_range').agg({
            'avg_power_w_mean': 'mean',
            'energy_per_token_mean': 'mean',
            'question_count': 'sum'
        }).to_dict('index')
        insights['power_by_token_range'] = power_by_range
        efficiency_rankings = self.get_model_efficiency_rankings()
        insights['model_efficiency_rankings'] = efficiency_rankings.to_dict('records')
        scaling_data = self.analysis_data.groupby('target_token_range')['avg_power_w_mean'].mean()
        insights['power_scaling_trend'] = {
            'min_power': float(scaling_data.min()),
            'max_power': float(scaling_data.max()),
            'power_range': float(scaling_data.max() - scaling_data.min()),
            'average_power': float(scaling_data.mean())
        }
        return insights
    
    def save_insights_data(self, output_path: str) -> str:
        """Save analysis data and insights to Excel file."""
        if self.analysis_data.empty:
            return ""
        print(f"Saving insights data to: {output_path}")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.analysis_data.to_excel(writer, sheet_name='Power_Analysis', index=False)
            efficiency_rankings = self.get_model_efficiency_rankings()
            efficiency_rankings.to_excel(writer, sheet_name='Model_Efficiency', index=False)
            insights = self.get_power_scaling_insights()
            power_by_range = pd.DataFrame.from_dict(
                insights.get('power_by_token_range', {}), 
                orient='index'
            )
            if not power_by_range.empty:
                power_by_range.to_excel(writer, sheet_name='Power_By_Range')
        return output_path 
