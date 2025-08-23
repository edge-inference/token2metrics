"""
Energy and Visualization 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from scipy.optimize import curve_fit
from .correlator import EnergyCorrelator

FIGURE_WIDTH = 4
FIGURE_HEIGHT = 4
FONT_SIZE_LABELS = 14      
FONT_SIZE_LEGEND = 12      
FONT_SIZE_TICKS = 12       
DPI = 300                 


class EnergyAnalyzer:
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette('colorblind')
        self.figure_config = {
            'width': FIGURE_WIDTH, 
            'height': FIGURE_HEIGHT,
            'dpi': DPI
        }
        
        # Consistent color mapping for models
        self.model_colors = {
            'DSR1-Qwen-1.5B': '#2ca02c',   # Green for 1.5B
            'DSR1-Llama-8B': '#ff7f0e',    # Gold/Orange for 8B  
            'DSR1-Qwen-8B': '#ff7f0e',     # Gold/Orange for 8B (alternative name)
            'DSR1-Qwen-14B': '#1f77b4'     # Blue for 14B
        }
    
    @staticmethod
    def logarithmic_function(x, a, b):
        return a * np.log(x) + b
    
    @staticmethod
    def negative_exponent_function(x, a, b):
        return a * np.power(x, -b)
    
    @staticmethod
    def power_function(x, a, b):
        return a * np.power(x, b)
    
    @staticmethod
    def exponential_function(x, a, b):
        return a * np.exp(b * x)
    
    def fit_curve(self, x_data: np.ndarray, y_data: np.ndarray, 
                  function_type: str, model_name: str, metric_name: str) -> Optional[Dict]:
        try:
            if function_type == 'logarithmic':
                if np.any(x_data <= 0):
                    print(f"⚠️  Cannot fit logarithmic for {model_name} {metric_name}: contains zero/negative x values")
                    return None
                func = self.logarithmic_function
                p0 = [1, 0]
                x_smooth = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
            elif function_type == 'negative_exponent':
                if np.any(x_data <= 0):
                    print(f"⚠️  Cannot fit negative_exponent for {model_name} {metric_name}: contains zero/negative x values")
                    return None
                func = self.negative_exponent_function
                p0 = [1, 1]
                x_smooth = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
            elif function_type == 'power':
                if np.any(x_data <= 0) or np.any(y_data <= 0):
                    print(f"⚠️  Cannot fit power for {model_name} {metric_name}: contains zero/negative values")
                    return None
                func = self.power_function
                p0 = [1, 1]
                x_smooth = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
            elif function_type == 'exponential':
                if np.any(y_data <= 0):
                    print(f"⚠️  Cannot fit exponential for {model_name} {metric_name}: contains zero/negative y values")
                    return None
                func = self.exponential_function
                p0 = [1, 0.1]
                x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
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
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'function': func
            }
        
        except Exception as e:
            print(f"⚠️  Failed to fit {function_type} for {model_name} {metric_name}: {e}")
            return None
    
    def run_analysis(self, energy_dir: str, performance_file: str) -> Dict[str, any]:
        print("🔋 Figure 6 Energy Analysis")
        print("=" * 40)
        
        correlator = EnergyCorrelator(energy_dir, performance_file)
        
        # Use MEAN-based correlation instead of seed-by-seed matching
        energy_loaded, perf_loaded = correlator.load_data_from_means()
        
        if not energy_loaded or not perf_loaded:
            print("❌ Failed to load data")
            return {}
        
        df = correlator.correlate_data()
        if df.empty:
            print("❌ No correlations found")
            return {}
        
        Path("results/energy").mkdir(parents=True, exist_ok=True)
        corr_file = correlator.save_results(df, "results/energy/energy_correlations.xlsx")
        
        insights = self._generate_insights(df)
        plots = self._create_visualizations(df)
        
        self._save_insights(insights)
        
        excel_file = self._export_plot_data_to_excel(df)
        
        print(f"\n✅ Analysis complete! Results in {self.output_dir}")
        return {
            'correlations': df,
            'insights': insights,
            'plots': plots,
            'correlation_file': corr_file,
            'excel_file': excel_file
        }
    
    def _generate_insights(self, df: pd.DataFrame) -> Dict[str, any]:
        insights = {}
        model_efficiency = df.groupby('model_name').agg({
            'avg_power_w': 'mean',
            'energy_per_decode_j': 'mean',
            'throughput_per_watt': 'mean',
            'decode_latency_s': 'mean'
        }).round(3)
        
        model_efficiency['efficiency_rank'] = model_efficiency['energy_per_decode_j'].rank()
        insights['model_efficiency'] = model_efficiency.to_dict('index')
        
        ps_scaling = df.groupby('ps_factor').agg({
            'avg_power_w': 'mean',
            'decode_latency_s': 'mean',
            'energy_per_decode_j': 'mean',
            'throughput': 'mean'
        }).round(3)
        insights['ps_scaling'] = ps_scaling.to_dict('index')
        
        power_scaling = {}
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            if len(model_data) > 1:
                min_ps = model_data['ps_factor'].min()
                max_ps = model_data['ps_factor'].max()
                
                min_power = model_data[model_data['ps_factor'] == min_ps]['avg_power_w'].mean()
                max_power = model_data[model_data['ps_factor'] == max_ps]['avg_power_w'].mean()
                
                power_scaling[model] = {
                    'ps_range': f"{min_ps}-{max_ps}",
                    'power_increase_factor': round(max_power / min_power if min_power > 0 else 0, 2),
                    'energy_efficiency': round(model_data['energy_per_decode_j'].mean(), 3)
                }
        
        insights['power_scaling'] = power_scaling
        
        curve_fitting_results = {}
        
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            model_results = {}
            
            if len(model_data) > 1:
                model_summary = model_data.groupby('ps_factor').agg({
                    'decode_latency_s': 'mean',
                    'energy_per_decode_j': 'mean',
                    'energy_per_sample_j': 'mean',
                    'avg_power_w': 'mean'
                })
                
                x_data = model_summary.index.values.astype(float)
                
                if 'decode_latency_s' in model_summary.columns:
                    y_data = model_summary['decode_latency_s'].values
                    fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'decode_latency')
                    if fit_result:
                        model_results['decode_latency_scaling'] = {
                            'function_type': fit_result['function_type'],
                            'r2_score': round(fit_result['r2_score'], 4),
                            'parameters': [round(p, 6) for p in fit_result['parameters']]
                        }
                    else:
                        model_results['decode_latency_scaling'] = {
                            'function_type': 'none',
                            'r2_score': 0.0,
                            'parameters': []
                        }
                
                if 'energy_per_decode_j' in model_summary.columns:
                    y_data = model_summary['energy_per_decode_j'].values
                    fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'energy_per_decode')
                    if fit_result:
                        model_results['energy_per_decode_scaling'] = {
                            'function_type': fit_result['function_type'],
                            'r2_score': round(fit_result['r2_score'], 4),
                            'parameters': [round(p, 6) for p in fit_result['parameters']]
                        }
                    else:
                        model_results['energy_per_decode_scaling'] = {
                            'function_type': 'none',
                            'r2_score': 0.0,
                            'parameters': []
                        }
                
                if 'energy_per_sample_j' in model_summary.columns:
                    y_data = model_summary['energy_per_sample_j'].values
                    fit_result = self.fit_curve(x_data, y_data, 'negative_exponent', model, 'energy_per_sample')
                    if fit_result:
                        model_results['energy_per_sample_scaling'] = {
                            'function_type': fit_result['function_type'],
                            'r2_score': round(fit_result['r2_score'], 4),
                            'parameters': [round(p, 6) for p in fit_result['parameters']]
                        }
                    else:
                        model_results['energy_per_sample_scaling'] = {
                            'function_type': 'none',
                            'r2_score': 0.0,
                            'parameters': []
                        }
                
                if 'avg_power_w' in model_summary.columns:
                    y_data = model_summary['avg_power_w'].values
                    fit_result = self.fit_curve(x_data, y_data, 'exponential', model, 'avg_power')
                    if fit_result:
                        model_results['power_scaling'] = {
                            'function_type': fit_result['function_type'],
                            'r2_score': round(fit_result['r2_score'], 4),
                            'parameters': [round(p, 6) for p in fit_result['parameters']]
                        }
                    else:
                        model_results['power_scaling'] = {
                            'function_type': 'none',
                            'r2_score': 0.0,
                            'parameters': []
                        }
            
            curve_fitting_results[model] = model_results
        
        insights['curve_fitting_results'] = curve_fitting_results
        
        best_efficiency = df.loc[df['energy_per_decode_j'].idxmin()]
        best_throughput = df.loc[df['throughput'].idxmax()]
        
        insights['best_configs'] = {
            'most_efficient': {
                'model': best_efficiency['model_name'],
                'ps_factor': best_efficiency['ps_factor'],
                'energy_per_decode': best_efficiency['energy_per_decode_j']
            },
            'highest_throughput': {
                'model': best_throughput['model_name'],
                'ps_factor': best_throughput['ps_factor'],
                'throughput': best_throughput['throughput']
            }
        }
        
        return insights
    
    def _shorten_model_name(self, model_name: str) -> str:
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

    def _create_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        plots = {}
        
        plots['decode_latency_vs_ps'] = self._plot_decode_latency_vs_ps(df)
        plots['energy_per_decode_vs_ps'] = self._plot_energy_per_decode_vs_ps(df)
        plots['energy_per_sample_vs_ps'] = self._plot_energy_per_sample_vs_ps(df)
        plots['power_vs_ps'] = self._plot_power_vs_ps(df)
        plots['latency_energy_tradeoff'] = self._plot_latency_energy_tradeoff(df)
        
        # Add system metrics plots if data is available
        if 'avg_ram_usage_pct' in df.columns:
            plots['ram_usage_vs_ps'] = self._plot_ram_usage_vs_ps(df)
        if 'avg_gpu_usage_pct' in df.columns:
            plots['gpu_usage_vs_ps'] = self._plot_gpu_usage_vs_ps(df)
        if 'avg_temp_tj_c' in df.columns:
            plots['temperature_vs_ps'] = self._plot_temperature_vs_ps(df)
        
        return plots
    
    def _plot_decode_latency_vs_ps(self, df: pd.DataFrame) -> str:
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
    
    def _plot_energy_per_decode_vs_ps(self, df: pd.DataFrame) -> str:
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

    def _plot_energy_per_sample_vs_ps(self, df: pd.DataFrame) -> str:
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
    
    def _plot_power_vs_ps(self, df: pd.DataFrame) -> str:
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
    
    def _plot_efficiency_comparison(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        model_efficiency = df.groupby('model_name')['throughput_per_watt'].mean().sort_values(ascending=True)
        
        short_names = [self._shorten_model_name(name) for name in model_efficiency.index]
        
        bars = ax.barh(range(len(model_efficiency)), model_efficiency.values)
        ax.set_yticks(range(len(model_efficiency)))
        ax.set_yticklabels(short_names, fontsize=FONT_SIZE_TICKS)
        ax.set_xlabel('Throughput per Watt', fontsize=FONT_SIZE_LABELS)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        colors = sns.color_palette('viridis', len(model_efficiency))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        output_path = self.output_dir / "efficiency_comparison.pdf"
        plt.savefig(output_path, dpi=self.figure_config['dpi'], bbox_inches='tight', format='pdf')
        plt.close()
        
        return str(output_path)
    
    def _plot_latency_energy_tradeoff(self, df: pd.DataFrame) -> str:
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
    
    def _plot_ram_usage_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot RAM usage percentage vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            
            # Skip if no RAM data for this model
            if 'avg_ram_usage_pct' not in model_data.columns or model_data['avg_ram_usage_pct'].isna().all():
                continue
                
            model_summary = model_data.groupby('ps_factor')['avg_ram_usage_pct'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='o')
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
    
    def _plot_gpu_usage_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot GPU usage percentage vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            
            # Skip if no GPU data for this model
            if 'avg_gpu_usage_pct' not in model_data.columns or model_data['avg_gpu_usage_pct'].isna().all():
                continue
                
            model_summary = model_data.groupby('ps_factor')['avg_gpu_usage_pct'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='s')
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
    
    def _plot_temperature_vs_ps(self, df: pd.DataFrame) -> str:
        """Plot temperature vs parallel scaling factor."""
        fig, ax = plt.subplots(figsize=(self.figure_config['width'], self.figure_config['height']))
        
        # Sort models by size (smaller first)
        sorted_models = self._sort_models_by_size(df['model_name'].unique())
        
        for model in sorted_models:
            color = self._get_model_color(model)
            model_data = df[df['model_name'] == model]
            
            # Skip if no temperature data for this model
            if 'avg_temp_tj_c' not in model_data.columns or model_data['avg_temp_tj_c'].isna().all():
                continue
                
            model_summary = model_data.groupby('ps_factor')['avg_temp_tj_c'].mean()
            short_name = self._shorten_model_name(model)
            
            x_data = model_summary.index.values.astype(float)
            y_data = model_summary.values
            
            ax.scatter(x_data, y_data, alpha=0.7, color=color, s=50, marker='^')
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
    
    def _save_insights(self, insights: Dict[str, any]) -> None:
        import json
        
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        insights_serializable = convert_types(insights)
        
        output_path = self.output_dir / "energy_insights.json"
        with open(output_path, 'w') as f:
            json.dump(insights_serializable, f, indent=2)
        
        print(f"Saved insights to {output_path}")
    
    def _export_plot_data_to_excel(self, df: pd.DataFrame) -> str:
        try:
            import openpyxl
        except ImportError:
            print("⚠️ openpyxl not available, using xlsxwriter instead")
            engine = 'xlsxwriter'
        else:
            engine = 'openpyxl'
        
        output_path = self.output_dir / "plot_data.xlsx"
        
        fit_results = {}
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            if len(model_data) > 1:
                model_summary = model_data.groupby('ps_factor').agg({
                    'decode_latency_s': 'mean',
                    'energy_per_decode_j': 'mean',
                    'energy_per_sample_j': 'mean',
                    'avg_power_w': 'mean'
                })
                
                x_data = model_summary.index.values.astype(float)
                
                latency_fit = self.fit_curve(x_data, model_summary['decode_latency_s'].values, 'exponential', model, 'decode_latency')
                energy_decode_fit = self.fit_curve(x_data, model_summary['energy_per_decode_j'].values, 'exponential', model, 'energy_per_decode')
                energy_sample_fit = self.fit_curve(x_data, model_summary['energy_per_sample_j'].values, 'negative_exponent', model, 'energy_per_sample')
                power_fit = self.fit_curve(x_data, model_summary['avg_power_w'].values, 'exponential', model, 'avg_power')
                
                fit_results[model] = {
                    'latency': latency_fit,
                    'energy_decode': energy_decode_fit,
                    'energy_sample': energy_sample_fit,
                    'power': power_fit,
                    'summary': model_summary
                }
        
        with pd.ExcelWriter(output_path, engine=engine) as writer:
            
            latency_data = []
            for model in df['model_name'].unique():
                short_name = self._shorten_model_name(model)
                
                if model in fit_results:
                    model_summary = fit_results[model]['summary']['decode_latency_s']
                    fit_result = fit_results[model]['latency']
                    
                    fit_type = fit_result['function_type'] if fit_result else 'none'
                    r2_score = fit_result['r2_score'] if fit_result else 0.0
                    parameters = fit_result['parameters'] if fit_result else []
                    
                    for ps_factor, latency_value in model_summary.items():
                        latency_data.append({
                            'Model': model,
                            'Short_Name': short_name,
                            'PS_Factor': ps_factor,
                            'Decode_Latency_s': latency_value,
                            'Fit_Type': fit_type,
                            'R2_Score': r2_score,
                            'Fit_Parameters': str(parameters)
                        })
            
            latency_df = pd.DataFrame(latency_data)
            latency_df.to_excel(writer, sheet_name='Decode_Latency_vs_PS', index=False)
            
            energy_decode_data = []
            for model in df['model_name'].unique():
                short_name = self._shorten_model_name(model)
                
                if model in fit_results:
                    model_summary = fit_results[model]['summary']['energy_per_decode_j']
                    fit_result = fit_results[model]['energy_decode']
                    
                    fit_type = fit_result['function_type'] if fit_result else 'none'
                    r2_score = fit_result['r2_score'] if fit_result else 0.0
                    parameters = fit_result['parameters'] if fit_result else []
                    
                    for ps_factor, energy_value in model_summary.items():
                        energy_decode_data.append({
                            'Model': model,
                            'Short_Name': short_name,
                            'PS_Factor': ps_factor,
                            'Energy_per_Decode_J': energy_value,
                            'Fit_Type': fit_type,
                            'R2_Score': r2_score,
                            'Fit_Parameters': str(parameters)
                        })
            
            energy_decode_df = pd.DataFrame(energy_decode_data)
            energy_decode_df.to_excel(writer, sheet_name='Energy_per_Decode_vs_PS', index=False)
            
            energy_sample_data = []
            for model in df['model_name'].unique():
                short_name = self._shorten_model_name(model)
                
                if model in fit_results:
                    model_summary = fit_results[model]['summary']['energy_per_sample_j']
                    fit_result = fit_results[model]['energy_sample']
                    
                    fit_type = fit_result['function_type'] if fit_result else 'none'
                    r2_score = fit_result['r2_score'] if fit_result else 0.0
                    parameters = fit_result['parameters'] if fit_result else []
                    
                    for ps_factor, energy_value in model_summary.items():
                        energy_sample_data.append({
                            'Model': model,
                            'Short_Name': short_name,
                            'PS_Factor': ps_factor,
                            'Energy_per_Sample_J': energy_value,
                            'Fit_Type': fit_type,
                            'R2_Score': r2_score,
                            'Fit_Parameters': str(parameters)
                        })
            
            energy_sample_df = pd.DataFrame(energy_sample_data)
            energy_sample_df.to_excel(writer, sheet_name='Energy_per_Sample_vs_PS', index=False)
            
            power_data = []
            for model in df['model_name'].unique():
                short_name = self._shorten_model_name(model)
                
                if model in fit_results:
                    model_summary = fit_results[model]['summary']['avg_power_w']
                    fit_result = fit_results[model]['power']
                    
                    fit_type = fit_result['function_type'] if fit_result else 'none'
                    r2_score = fit_result['r2_score'] if fit_result else 0.0
                    parameters = fit_result['parameters'] if fit_result else []
                    
                    for ps_factor, power_value in model_summary.items():
                        power_data.append({
                            'Model': model,
                            'Short_Name': short_name,
                            'PS_Factor': ps_factor,
                            'Avg_Power_W': power_value,
                            'Fit_Type': fit_type,
                            'R2_Score': r2_score,
                            'Fit_Parameters': str(parameters)
                        })
            
            power_df = pd.DataFrame(power_data)
            power_df.to_excel(writer, sheet_name='Power_vs_PS', index=False)
            
            # Add system metrics sheets if data is available
            system_metrics = ['avg_ram_usage_pct', 'avg_gpu_usage_pct', 'avg_temp_tj_c']
            for metric in system_metrics:
                if metric in df.columns and not df[metric].isna().all():
                    system_data = []
                    for model in df['model_name'].unique():
                        model_data = df[df['model_name'] == model]
                        short_name = self._shorten_model_name(model)
                        
                        if metric in model_data.columns and not model_data[metric].isna().all():
                            model_summary = model_data.groupby('ps_factor')[metric].mean()
                            
                            for ps_factor, metric_value in model_summary.items():
                                system_data.append({
                                    'Model': model,
                                    'Short_Name': short_name,
                                    'PS_Factor': ps_factor,
                                    f'{metric.replace("avg_", "").replace("_", " ").title()}': metric_value
                                })
                    
                    if system_data:
                        system_df = pd.DataFrame(system_data)
                        sheet_name = f'{metric.replace("avg_", "").replace("_", " ").title()}_vs_PS'
                        system_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            tradeoff_data = []
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                short_name = self._shorten_model_name(model)
                
                for _, row in model_data.iterrows():
                    tradeoff_record = {
                        'Model': model,
                        'Short_Name': short_name,
                        'PS_Factor': row['ps_factor'],
                        'Decode_Latency_s': row['decode_latency_s'],
                        'Energy_per_Decode_J': row['energy_per_decode_j'],
                        'Throughput': row.get('throughput', 0),
                        'Avg_Power_W': row['avg_power_w']
                    }
                    
                    # Add system metrics to tradeoff data if available
                    for metric in system_metrics:
                        if metric in row and pd.notna(row[metric]):
                            tradeoff_record[f'{metric.replace("avg_", "").replace("_", " ").title()}'] = row[metric]
                    
                    tradeoff_data.append(tradeoff_record)
            
            tradeoff_df = pd.DataFrame(tradeoff_data)
            tradeoff_df.to_excel(writer, sheet_name='Latency_Energy_Tradeoff', index=False)
            
            efficiency_data = []
            model_efficiency = df.groupby('model_name')['throughput_per_watt'].mean().sort_values(ascending=True)
            
            for model, throughput_per_watt in model_efficiency.items():
                short_name = self._shorten_model_name(model)
                model_data = df[df['model_name'] == model]
                
                efficiency_record = {
                    'Model': model,
                    'Short_Name': short_name,
                    'Throughput_per_Watt': throughput_per_watt,
                    'Avg_Energy_per_Decode_J': model_data['energy_per_decode_j'].mean(),
                    'Avg_Power_W': model_data['avg_power_w'].mean(),
                    'Avg_Throughput': model_data.get('throughput', pd.Series([0])).mean()
                }
                
                # Add system metrics to efficiency data if available
                for metric in system_metrics:
                    if metric in model_data.columns and not model_data[metric].isna().all():
                        efficiency_record[f'Avg_{metric.replace("avg_", "").replace("_", " ").title()}'] = model_data[metric].mean()
                
                efficiency_data.append(efficiency_record)
            
            efficiency_df = pd.DataFrame(efficiency_data)
            efficiency_df.to_excel(writer, sheet_name='Efficiency_Comparison', index=False)
            
            summary_data = df.copy()
            summary_data['Short_Name'] = summary_data['model_name'].apply(self._shorten_model_name)
            summary_data.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        print(f"✅ Exported plot data to Excel: {output_path}")
        return str(output_path)


def main():
    """Test the energy analyzer."""
    analyzer = EnergyAnalyzer()
    
    results = analyzer.run_analysis(
        energy_dir="../tegra/figure6",
        performance_file="results/energy/excel_scaling_summary.xlsx"
    )
    
    if results:
        print(f"\nGenerated {len(results['plots'])} plots")
        print(f"Analyzed {len(results['correlations'])} data points")
        print(f"Found {len(results['insights']['model_efficiency'])} models")


if __name__ == "__main__":
    main() 