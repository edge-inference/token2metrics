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
Figure 6 Token Analyzer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
from scipy.optimize import curve_fit
import json
import glob

FIGURE_WIDTH = 4
FIGURE_HEIGHT = 4
FONT_SIZE_LABELS = 14      
FONT_SIZE_LEGEND = 12      
FONT_SIZE_TICKS = 12       
SHOW_R2_IN_LEGEND = False  


class TokenAnalyzer:
    
    def __init__(self, excel_dir: str = "results/tokens"):
        self.excel_dir = Path(excel_dir)
        
        self.model_display_names = {
            'DeepSeek-R1-Distill-Qwen-1.5B': 'DSR1-Qwen-1.5B',
            'DeepSeek-R1-Distill-Qwen-14B': 'DSR1-Qwen-14B',
            'DeepSeek-R1-Distill-Llama-8B': 'DSR1-Llama-8B'
        }
        
        # Consistent color mapping for models
        self.model_colors = {
            'DSR1-Qwen-1.5B': '#2ca02c',   # Green for 1.5B
            'DSR1-Llama-8B': '#ff7f0e',    # Gold/Orange for 8B  
            'DSR1-Qwen-8B': '#ff7f0e',     # Gold/Orange for 8B (alternative name)
            'DSR1-Qwen-14B': '#1f77b4'     # Blue for 14B
        }
    
    @staticmethod
    def linear_function(x, a, b):
        return a * x + b
    
    @staticmethod
    def power_function(x, a, b):
        return a * np.power(x, b)
    
    @staticmethod
    def log_function(x, a, b):
        return a * np.log(x) + b
    
    @staticmethod
    def exponential_function(x, a, b):
        return a * np.exp(b * x)
    
    def find_excel_files(self) -> List[Path]:
        """Find all Excel files in the directory, excluding summary files."""
        if not self.excel_dir.exists():
            print(f"❌ Excel directory {self.excel_dir} does not exist")
            return []
        
        excel_files = []
        for file_path in self.excel_dir.glob("*.xlsx"):
            if 'summary' in file_path.name.lower():
                print(f"📄 Skipping summary file: {file_path.name}")
                continue
            excel_files.append(file_path)
        
        return sorted(excel_files)
    
    def read_excel_data(self, excel_file: Path) -> pd.DataFrame:
        """Read data from all sheets in an Excel file."""
        print(f"📊 Reading {excel_file.name}")
        
        all_data = []
        
        try:
            excel_file_obj = pd.ExcelFile(excel_file)
            sheet_names = excel_file_obj.sheet_names
            
            for sheet_name in sheet_names:
                try:
                    ps_factor = int(sheet_name.split('_')[0])
                except:
                    print(f"⚠️  Skipping sheet {sheet_name} - cannot parse sample count")
                    continue
                
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                data_df = df[~df.iloc[:, 0].astype(str).isin(['MEAN', 'STD', 'MIN', 'MAX'])].copy()
                
                if not data_df.empty:
                    data_df['ps_factor'] = ps_factor
                    all_data.append(data_df)
        
        except Exception as e:
            print(f"❌ Error reading {excel_file.name}: {e}")
            return pd.DataFrame()
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"   ✅ Read {len(combined_df)} data points across {len(all_data)} PS factors")
            return combined_df
        else:
            print(f"   ❌ No valid data found in {excel_file.name}")
            return pd.DataFrame()
    
    def extract_model_name(self, excel_file: Path) -> str:
        """Extract model name from Excel filename."""
        model_name = excel_file.stem
        return self.model_display_names.get(model_name, model_name)
    
    def load_all_excel_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from all Excel files."""
        excel_files = self.find_excel_files()
        
        if not excel_files:
            print("❌ No Excel files found")
            return {}
        
        print(f"📁 Found {len(excel_files)} Excel files")
        
        all_model_data = {}
        
        for excel_file in excel_files:
            model_name = self.extract_model_name(excel_file)
            model_data = self.read_excel_data(excel_file)
            
            if not model_data.empty:
                all_model_data[model_name] = model_data
            
        return all_model_data
    
    def aggregate_by_ps_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by PS factor (sample count) across seeds."""
        agg_data = []
        
        for ps_factor in sorted(df['ps_factor'].unique()):
            ps_data = df[df['ps_factor'] == ps_factor]
            
            agg_row = {
                'ps_factor': ps_factor,
                'num_runs': len(ps_data)
            }
            
            numeric_cols = ps_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'ps_factor':
                    agg_row[f'{col}_mean'] = ps_data[col].mean()
                    agg_row[f'{col}_std'] = ps_data[col].std()
            
            agg_data.append(agg_row)
        
        return pd.DataFrame(agg_data)
    
    def fit_scaling_curve(self, x_data: np.ndarray, y_data: np.ndarray, 
                         model_name: str, metric_name: str) -> Dict[str, Any]:
        """Fit scaling curve using natural exponential function."""
        try:
            if np.any(y_data <= 0):
                print(f"⚠️  Cannot fit exponential for {model_name} {metric_name}: contains zero/negative y values")
                return None
            
            popt, pcov = curve_fit(self.exponential_function, x_data, y_data, p0=[1, 0.1], maxfev=5000)
            y_pred = self.exponential_function(x_data, *popt)
            
            # Calculate R²
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            y_smooth = self.exponential_function(x_smooth, *popt)
            
            return {
                'function_name': f'{model_name}_{metric_name}_exponential',
                'function_type': 'exponential',
                'parameters': popt.tolist(),
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'function': self.exponential_function
            }
        
        except Exception as e:
            print(f"⚠️  Failed to fit exponential function for {model_name} {metric_name}: {e}")
            return None
    
    def analyze_token_scaling(self) -> Dict[str, Any]:
        """
        Analyze token-related scaling patterns across all models.
        
        Returns:
            Dictionary with analysis results for each model
        """
        all_model_data = self.load_all_excel_data()
        
        if not all_model_data:
            print("❌ No data loaded for analysis")
            return {}
        
        print(f"\n🔍 Analyzing token scaling for {len(all_model_data)} models...")
        
        analysis_results = {}
        
        for model_name, model_data in all_model_data.items():
            print(f"\n📊 Analyzing {model_name}...")
            
            agg_data = self.aggregate_by_ps_factor(model_data)
            ps_factors = agg_data['ps_factor'].values
            
            latency_fit = None
            if 'avg_decode_time_mean' in agg_data.columns:
                latency_data = agg_data['avg_decode_time_mean'].values
                latency_fit = self.fit_scaling_curve(
                    ps_factors, latency_data, model_name, 'latency'
                )
                if latency_fit:
                    print(f"    Latency: {latency_fit['function_type']} (R² = {latency_fit['r2_score']:.3f})")
                else:
                    print(f"    Latency: Failed to fit exponential curve")
            else:
                print(f"    Latency: No data available")
            
            analysis_results[model_name] = {
                'latency_scaling': latency_fit,
                'data': agg_data,
                'ps_factors': ps_factors
            }
        
        return analysis_results
    
    def _shorten_model_name(self, model_name: str) -> str:
        """Convert model name to DSR1-{modelname}-{size} format."""
        if 'Qwen-1.5B' in model_name:
            return 'DSR1-Qwen-1.5B'
        elif 'Qwen-14B' in model_name:
            return 'DSR1-Qwen-14B'
        elif 'Llama-8B' in model_name:
            return 'DSR1-Llama-8B'
        else:
            parts = model_name.split('-')
            if len(parts) >= 2:
                return f"DSR1-{parts[-2]}-{parts[-1]}"
            return f"DSR1-{model_name}"

    def create_scaling_plots(self, analysis_results: Dict[str, Any], 
                           output_dir: str = "scaling_analysis"):
        """Create scaling plots from Excel-based analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Sort models by size (smaller first) and use consistent colors
        sorted_models = self._sort_models_by_size(list(analysis_results.keys()))
        
        # --- Decode Latency vs PS Plot ---
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        for model_name in sorted_models:
            color = self._get_model_color(model_name)
            result = analysis_results[model_name]
            latency_fit = result['latency_scaling']
            
            if latency_fit and 'x_data' in latency_fit:
                x_data = latency_fit['x_data']
                y_data = latency_fit['y_data']
                r2_score = latency_fit['r2_score']
                
                ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50)
                
                short_name = self._shorten_model_name(model_name)
                if SHOW_R2_IN_LEGEND:
                    legend_label = f'{short_name} (R² = {r2_score:.3f})'
                else:
                    legend_label = short_name
                
                if 'x_smooth' in latency_fit and 'y_smooth' in latency_fit:
                    ax.plot(latency_fit['x_smooth'], latency_fit['y_smooth'], '--',
                           color=color, linewidth=2, label=legend_label)
                else:
                    y_pred = latency_fit['y_pred']
                    sort_idx = np.argsort(x_data)
                    ax.plot(x_data[sort_idx], y_pred[sort_idx], '--',
                           color=color, linewidth=2, label=legend_label)
        
        ax.set_xlabel('SF (Parallel Scaling Factor)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Decode Latency (s)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        latency_plot_path = output_path / 'decode_latency_vs_ps.pdf'
        plt.savefig(latency_plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved decode latency scaling plot: {latency_plot_path}")
    
    def save_analysis_summary(self, analysis_results: Dict[str, Any], 
                             output_dir: str = "scaling_analysis"):
        """Save analysis summary to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        summary = {}
        
        for model_name, result in analysis_results.items():
            latency_fit = result['latency_scaling']
            
            summary[model_name] = {
                'latency_scaling': {
                    'function_type': latency_fit.get('function_type', 'none') if latency_fit else 'none',
                    'r2_score': latency_fit.get('r2_score', 0.0) if latency_fit else 0.0,
                    'parameters': latency_fit.get('parameters', []) if latency_fit else []
                }
            }
        
        summary_path = output_path / 'excel_scaling_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved analysis summary: {summary_path}")
    
    def print_scaling_summary(self, analysis_results: Dict[str, Any]):
        """Print scaling analysis summary."""
        print("\n" + "="*60)
        print("🔋 FIGURE 6: EXCEL-BASED SCALING ANALYSIS")
        print("="*60)
        
        for model_name, result in analysis_results.items():
            latency_fit = result['latency_scaling']
            
            print(f"\n{model_name}:")
            if latency_fit:
                print(f"  Latency vs PS: {latency_fit['function_type']} (R² = {latency_fit['r2_score']:.4f})")
            else:
                print(f"  Latency vs PS: No data available")
        
        print("\n" + "="*60)

    def _get_model_sort_key(self, model_name: str) -> int:
        """Get sort key for consistent model ordering (smaller models first)."""
        short_name = self._shorten_model_name(model_name)
        if '1.5B' in short_name:
            return 0  # 1.5B first
        elif '8B' in short_name:
            return 1  # 8B second
        elif '14B' in short_name:
            return 2  # 14B third
        else:
            return 999  # Unknown models last
    
    def _get_model_color(self, model_name: str) -> str:
        """Get consistent color for model."""
        short_name = self._shorten_model_name(model_name)
        return self.model_colors.get(short_name, '#d62728')  # Default red if not found
    
    def _sort_models_by_size(self, model_list: list) -> list:
        """Sort models by size (smaller first)."""
        return sorted(model_list, key=self._get_model_sort_key) 