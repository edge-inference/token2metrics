#!/usr/bin/env python3
"""
Standalone Energy Plotting Script
Reads from energy_correlations.xlsx and generates all energy plots.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from typing import Optional, Dict

# Plot styling
plt.style.use('default')
sns.set_palette("husl")

FONT_SIZE_TITLE = 14
FONT_SIZE_LABELS = 12
FONT_SIZE_LEGEND = 10  
FONT_SIZE_TICKS = 10

class EnergyPlotter:
    """Standalone energy plotter that reads from correlation Excel file."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figure_config = {
            'width': 4,
            'height': 4,
            'dpi': 300
        }
        
        # Consistent color mapping for models
        self.model_colors = {
            'DSR1-Qwen-1.5B': '#2ca02c',   # Green for 1.5B
            'DSR1-Llama-8B': '#ff7f0e',    # Gold/Orange for 8B  
            'DSR1-Qwen-8B': '#ff7f0e',     # Gold/Orange for 8B (alternative name)
            'DSR1-Qwen-14B': '#1f77b4'     # Blue for 14B
        }
    
    @staticmethod
    def exponential_function(x, a, b):
        return a * np.exp(b * x)
    
    @staticmethod
    def negative_exponent_function(x, a, b):
        return a * (x ** (-b))
    
    def fit_curve(self, x_data: np.ndarray, y_data: np.ndarray, 
                  function_type: str, model_name: str, metric_name: str) -> Optional[Dict]:
        """Fit curves to data"""
        try:
            if len(x_data) < 2:
                print(f"⚠️  Cannot fit {function_type} for {model_name} {metric_name}: Need at least 2 data points, got {len(x_data)}")
                return None
            
            if function_type == 'exponential':
                if np.any(y_data <= 0):
                    print(f"⚠️  Cannot fit exponential for {model_name} {metric_name}: contains zero/negative y values")
                    return None
                func = self.exponential_function
                p0 = [1, 0.1]
                x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            elif function_type == 'negative_exponent':
                if np.any(x_data <= 0):
                    print(f"⚠️  Cannot fit negative_exponent for {model_name} {metric_name}: contains zero/negative x values")
                    return None
                func = self.negative_exponent_function
                p0 = [1, 1]
                x_smooth = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
            else:
                print(f"⚠️  Unknown function type: {function_type}")
                return None
            
            popt, pcov = curve_fit(func, x_data, y_data, p0=p0, maxfev=5000)
            y_pred = func(x_data, *popt)
            
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            y_smooth = func(x_smooth, *popt)
            
            return {
                'function_type': function_type,
                'parameters': popt.tolist(),
                'r2_score': r2,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth
            }
        
        except Exception as e:
            print(f"⚠️  Failed to fit {function_type} for {model_name} {metric_name}: {e}")
            return None
    
    def _shorten_model_name(self, model_name: str) -> str:
        """Create short model names for plotting."""
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
    
    def load_correlation_data(self, excel_file: str) -> pd.DataFrame:
        """Load correlation data from Excel file."""
        try:
            df = pd.read_excel(excel_file, sheet_name='Correlations')
            print(f"✅ Loaded {len(df)} correlations from {excel_file}")
            print(f"   Models: {df['model_name'].unique().tolist()}")
            print(f"   PS factors: {sorted(df['ps_factor'].unique())}")
            return df
        except Exception as e:
            print(f"❌ Error loading {excel_file}: {e}")
            return pd.DataFrame()
    
    def plot_decode_latency_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot decode latency vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            model_summary = model_data.groupby('ps_factor')['decode_latency_s'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50)
            
            fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'decode_latency')
            
            if fit_result:
                ax.plot(fit_result['x_smooth'], fit_result['y_smooth'], '--',
                       color=color, linewidth=2, label=short_name)
                print(f"  {short_name} decode latency: exponential fit R² = {fit_result['r2_score']:.3f}")
            else:
                ax.plot(x_data, y_data, '--', color=color, linewidth=2, label=short_name)
        
        ax.set_xlabel('Parallel Scaling Factor (SF)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Decode Latency (s)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        output_path = self.output_dir / "decode_latency_vs_ps.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def plot_energy_per_decode_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot energy per decode vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            model_summary = model_data.groupby('ps_factor')['energy_per_decode_j'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50)
            
            fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'energy_per_decode')
            
            if fit_result:
                ax.plot(fit_result['x_smooth'], fit_result['y_smooth'], '--',
                       color=color, linewidth=2, label=short_name)
                print(f"  {short_name} energy/decode: exponential fit R² = {fit_result['r2_score']:.3f}")
            else:
                ax.plot(x_data, y_data, '--', color=color, linewidth=2, label=short_name)
        
        ax.set_xlabel('Parallel Scaling Factor (SF)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy (J/question)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        output_path = self.output_dir / "energy_per_decode_vs_ps.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)

    def plot_energy_per_sample_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot energy per sample vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            model_summary = model_data.groupby('ps_factor')['energy_per_sample_j'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='s')
            
            fit_result = self.fit_curve(x_data, y_data, 'negative_exponent', model, 'energy_per_sample')
            
            if fit_result:
                ax.plot(fit_result['x_smooth'], fit_result['y_smooth'], '--',
                       color=color, linewidth=2, label=short_name)
                print(f"  {short_name} energy/sample: negative exponent fit R² = {fit_result['r2_score']:.3f}")
            else:
                ax.plot(x_data, y_data, '--', color=color, linewidth=2, label=short_name)
        
        ax.set_xlabel('Parallel Scaling Factor (SF)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy (J/SF)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        output_path = self.output_dir / "energy_per_sample_vs_ps.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def plot_power_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot average power vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            model_summary = model_data.groupby('ps_factor')['avg_power_w'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='s')
            
            fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'avg_power')
            
            if fit_result:
                ax.plot(fit_result['x_smooth'], fit_result['y_smooth'], '--',
                       color=color, linewidth=2, label=short_name)
                print(f"  {short_name} power: exponential fit R² = {fit_result['r2_score']:.3f}")
            else:
                ax.plot(x_data, y_data, '--', color=color, linewidth=2, label=short_name)
        
        ax.set_xlabel('Parallel Scaling Factor (SF)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Average Power (W)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        output_path = self.output_dir / "power_vs_ps.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def plot_latency_energy_tradeoff(self, df: pd.DataFrame) -> str:
        """Plot latency vs energy tradeoff scatter plot."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            short_name = self._shorten_model_name(model)
            ax.scatter(model_data['decode_latency_s'], model_data['energy_per_decode_j'],
                      label=short_name, alpha=0.7, s=60, color=color)
        
        ax.set_xlabel('Decode Latency (s)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy per Decode (J)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        plt.tight_layout()
        output_path = self.output_dir / "latency_energy_tradeoff.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def plot_ram_usage_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot RAM usage vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            model_summary = model_data.groupby('ps_factor')['avg_ram_usage_pct'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='s')
            
            fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'avg_ram_usage')
            
            if fit_result:
                ax.plot(fit_result['x_smooth'], fit_result['y_smooth'], '--',
                       color=color, linewidth=2, label=short_name)
                print(f"  {short_name} RAM usage: exponential fit R² = {fit_result['r2_score']:.3f}")
            else:
                ax.plot(x_data, y_data, '--', color=color, linewidth=2, label=short_name)
        
        ax.set_xlabel('Parallel Scaling Factor (SF)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('RAM Usage (%)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        output_path = self.output_dir / "ram_usage_vs_ps.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def plot_gpu_usage_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot GPU usage vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            model_summary = model_data.groupby('ps_factor')['avg_gpu_usage_pct'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='s')
            
            fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'avg_gpu_usage')
            
            if fit_result:
                ax.plot(fit_result['x_smooth'], fit_result['y_smooth'], '--',
                       color=color, linewidth=2, label=short_name)
                print(f"  {short_name} GPU usage: exponential fit R² = {fit_result['r2_score']:.3f}")
            else:
                ax.plot(x_data, y_data, '--', color=color, linewidth=2, label=short_name)
        
        ax.set_xlabel('Parallel Scaling Factor (SF)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('GPU Usage (%)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        output_path = self.output_dir / "gpu_usage_vs_ps.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def plot_temperature_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot temperature vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            model_summary = model_data.groupby('ps_factor')['avg_temp_tj_c'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='s')
            
            fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'avg_temp')
            
            if fit_result:
                ax.plot(fit_result['x_smooth'], fit_result['y_smooth'], '--',
                       color=color, linewidth=2, label=short_name)
                print(f"  {short_name} temperature: exponential fit R² = {fit_result['r2_score']:.3f}")
            else:
                ax.plot(x_data, y_data, '--', color=color, linewidth=2, label=short_name)
        
        ax.set_xlabel('Parallel Scaling Factor (SF)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Temperature (°C)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        plt.tight_layout()
        output_path = self.output_dir / "temperature_vs_ps.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def plot_dual_axis_ps_power_gpu(self, df: pd.DataFrame) -> str:
        """Plot dual-axis: Parallel Scaling Factor vs Power & GPU Utilization."""
        fig, ax1 = plt.subplots(figsize=(5, 5))
        
        # Create second y-axis
        ax2 = ax1.twinx()
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            short_name = self._shorten_model_name(model)
            
            # Filter out rows with missing GPU utilization data
            valid_data = model_data.dropna(subset=['avg_gpu_usage_pct'])
            
            if len(valid_data) == 0:
                print(f"⚠️  No valid GPU utilization data for {short_name}")
                continue
            
            # Group by PS factor and take mean (in case of multiple runs)
            grouped = valid_data.groupby('ps_factor').agg({
                'avg_power_w': 'mean',
                'avg_gpu_usage_pct': 'mean'
            }).reset_index()
            
            x_data = grouped['ps_factor'].values.astype(float)
            power_data = grouped['avg_power_w'].values
            gpu_data = grouped['avg_gpu_usage_pct'].values
            
            # Plot power on left axis (solid lines, circles)
            ax1.plot(x_data, power_data, 'o-', color=color, linewidth=2, 
                    markersize=6, alpha=0.9, markerfacecolor=color, 
                    markeredgecolor='white', markeredgewidth=1)
            
            # Plot GPU utilization on right axis (dashed lines, squares)
            ax2.plot(x_data, gpu_data, 's--', color=color, linewidth=2, 
                    markersize=6, alpha=0.9, markerfacecolor=color, 
                    markeredgecolor='white', markeredgewidth=1,
                    linestyle='--', dashes=(2, 6))
        
        # Configure left y-axis (Power)
        ax1.set_xlabel('Parallel Scaling Factor (SF)', fontsize=16)
        ax1.set_ylabel('Power (W)', fontsize=16, color='black')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        
        # Configure right y-axis (GPU Utilization)
        ax2.set_ylabel('GPU Utilization (%)', fontsize=16, color='#555555')
        ax2.tick_params(axis='y', labelcolor='#555555', labelsize=12)
        
        # Set log scale for x-axis
        ax1.set_xscale('log', base=2)
        ax1.set_xticks([1, 2, 4, 8, 16, 32])
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        # Create comprehensive legend combining models and line styles (like NCU parser)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Add Power/GPU line style indicators
        legend_elements.extend([
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, marker='o', 
                   markersize=4, label='Power (W)', markerfacecolor='black'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, marker='s', 
                   markersize=4, label='GPU Util (%)', markerfacecolor='black')
        ])
        
        # Add separator line
        legend_elements.append(Line2D([0], [0], color='none', alpha=0, label='─────────'))
        
        # Add model color indicators
        for model in sorted_models:
            color = self._get_model_color(model)
            short_name = self._shorten_model_name(model)
            legend_elements.append(
                Patch(facecolor=color, alpha=0.8, edgecolor='black', 
                      linewidth=0.5, label=short_name)
            )
        
        # Position legend below plot area to avoid squeezing chart
        ax1.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.20), loc='upper center', 
                  frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=13)
        
        # Clean styling - no grid, but keep top boundary that connects naturally
        ax1.spines['left'].set_color('black')
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['bottom'].set_color('black')
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['top'].set_visible(True)
        ax1.spines['top'].set_color('black')
        ax1.spines['top'].set_linewidth(1)
        ax1.spines['right'].set_visible(False)
        
        ax2.spines['right'].set_color('black')
        ax2.spines['right'].set_linewidth(1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # Clean white background
        ax1.set_facecolor('white')
        
        # Add some padding to make it look cleaner
        ax1.margins(x=0.02, y=0.05)
        ax2.margins(y=0.05)
        
        plt.tight_layout()
        output_path = self.output_dir / "dual_axis_ps_power_gpu.pdf"
        # Save with extra bottom padding for bottom legend (like NCU parser)
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', 
                   pad_inches=0.3, facecolor='white', format='pdf')
        plt.close()
        
        print(f"📊 Generated dual-axis plot: PS Factor vs Power & GPU Utilization")
        return str(output_path)
    
    def generate_all_plots(self, correlation_file: str) -> Dict[str, str]:
        """Generate all energy plots from correlation file."""
        df = self.load_correlation_data(correlation_file)
        
        if df.empty:
            print("❌ No data to plot")
            return {}
        
        plots = {}
        
        print("\n📈 Generating plots...")
        plots['decode_latency_vs_ps'] = self.plot_decode_latency_vs_ps(df)
        plots['energy_per_decode_vs_ps'] = self.plot_energy_per_decode_vs_ps(df)
        plots['energy_per_sample_vs_ps'] = self.plot_energy_per_sample_vs_ps(df)
        plots['power_vs_ps'] = self.plot_power_vs_ps(df)
        plots['latency_energy_tradeoff'] = self.plot_latency_energy_tradeoff(df)
        
        # Add system metrics plots if data is available
        if 'avg_ram_usage_pct' in df.columns and not df['avg_ram_usage_pct'].isna().all():
            plots['ram_usage_vs_ps'] = self.plot_ram_usage_vs_ps(df)
            print("📊 Generated RAM usage plot")
        
        if 'avg_gpu_usage_pct' in df.columns and not df['avg_gpu_usage_pct'].isna().all():
            plots['gpu_usage_vs_ps'] = self.plot_gpu_usage_vs_ps(df)
            print("📊 Generated GPU usage plot")
            
            # Add dual-axis plot combining PS factor, GPU utilization, and power
            plots['dual_axis_ps_power_gpu'] = self.plot_dual_axis_ps_power_gpu(df)
        
        if 'avg_temp_tj_c' in df.columns and not df['avg_temp_tj_c'].isna().all():
            plots['temperature_vs_ps'] = self.plot_temperature_vs_ps(df)
            print("📊 Generated temperature plot")
        
        return plots

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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate energy plots from correlation data')
    parser.add_argument('--input', '-i', 
                       default='results/energy/energy_correlations.xlsx',
                       help='Input correlation Excel file')
    parser.add_argument('--output', '-o', 
                       default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        print("Please run the energy analysis first to generate the correlation file.")
        return 1
    
    print("🔋 Standalone Energy Plotter")
    print("=" * 40)
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    
    plotter = EnergyPlotter(args.output)
    plots = plotter.generate_all_plots(args.input)
    
    if plots:
        print(f"\n✅ Generated {len(plots)} plots in {args.output}/")
        for plot_name, plot_path in plots.items():
            print(f"  📊 {plot_name}: {plot_path}")
        
        print(f"\n🎯 Summary:")
        print(f"  - All plots saved as PDF files")
        print(f"  - Ready to share with colleagues!")
        return 0
    else:
        print("❌ No plots generated")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 