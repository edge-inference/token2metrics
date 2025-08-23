"""
Main model fitting coordinator for power and energy trends.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import json

from .power_fits import PowerFitter
from .energy_fits import EnergyFitter

FIGURE_WIDTH = 4
FIGURE_HEIGHT = 4
FONT_SIZE_LABELS = 12
FONT_SIZE_LEGEND = 10
FONT_SIZE_TICKS = 11
SHOW_R2_IN_LEGEND = False

class ModelFitter:
    """
    Main class for fitting power and energy trends across different model sizes.
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.power_fitter = PowerFitter()
        self.energy_fitter = EnergyFitter()
        self.model_patterns = {
            '1.5B': ['1.5B', '1_5B', 'Qwen2.5-1.5B'],
            '8B': ['8B', '8_B', 'Qwen2.5-8B', 'Llama-3.1-8B'],
            '14B': ['14B', '14_B', 'DeepSeek-R1-Distill-Qwen-14B']
        }
        self.fitted_models = {}
        self.model_name_mapping = {
            'DeepSeek_R1_Distill_Llama_8B': 'DSR1-Llama-8B',
            'DeepSeek_R1_Distill_Qwen_1.5B': 'DSR1-Qwen-1.5B',
            'DeepSeek_R1_Distill_Qwen_1_5B': 'DSR1-Qwen-1.5B', 
            'DeepSeek_R1_Distill_Qwen_14B': 'DSR1-Qwen-14B'
        }
        self.size_mapping = {
            'DeepSeek_R1_Distill_Llama_8B': '8B',
            'DeepSeek_R1_Distill_Qwen_1.5B': '1.5B',
            'DeepSeek_R1_Distill_Qwen_1_5B': '1.5B', 
            'DeepSeek_R1_Distill_Qwen_14B': '14B'
        }
        self.display_to_size = {
            'DSR1-Llama-8B': '8B',
            'DSR1-Qwen-1.5B': '1.5B',
            'DSR1-Qwen-14B': '14B'
        }
        
    def detect_model_size(self, model_name: str) -> str:
        """Detect model size from model name."""
        if model_name in self.display_to_size:
            return self.display_to_size[model_name]
        if model_name in self.size_mapping:
            return self.size_mapping[model_name]
        model_name_upper = model_name.upper()
        if 'MAX' in model_name_upper:
            return 'ignore'
        elif '1.5' in model_name or '1_5' in model_name:
            return '1.5B'
        elif '8' in model_name and 'B' in model_name:
            return '8B'
        elif '14' in model_name and 'B' in model_name:
            return '14B'
        return 'unknown'
    
    def load_correlation_data(self, correlation_file: str) -> Dict[str, pd.DataFrame]:
        """
        Load data from energy_performance_correlation.xlsx sheets.
        
        Args:
            correlation_file: Path to the correlation Excel file
            
        Returns:
            Dictionary mapping model names to their DataFrames
        """
        excel_file = pd.ExcelFile(correlation_file)
        model_data = {}
        print(f"Loading data from {correlation_file}")
        print(f"Available sheets: {excel_file.sheet_names}")
        for sheet_name in excel_file.sheet_names:
            if sheet_name in ['Model_Summary', 'Subject_Analysis']:
                continue
            df = pd.read_excel(correlation_file, sheet_name=sheet_name)
            required_cols = ['output_tokens', 'avg_power_w']
            if all(col in df.columns for col in required_cols):
                display_name = self.model_name_mapping.get(sheet_name, sheet_name)
                model_data[display_name] = df
                print(f"  Loaded {display_name}: {len(df)} records")
            else:
                print(f"  Skipped {sheet_name}: missing required columns")
        return model_data
    
    def fit_model_data(self, df: pd.DataFrame, model_name: str) -> Dict:
        """
        Fit power and energy trends for a specific model using decode tokens.

        """
        model_size = self.detect_model_size(model_name)
        print(f"Fitting {model_name} (detected as {model_size})")
        if model_size == 'ignore':
            return None
        df_sorted = df.sort_values('output_tokens').copy()
        output_tokens = df_sorted['output_tokens'].values
        power_w = df_sorted['avg_power_w'].values
        power_results = self.power_fitter.fit_power_trend(
            output_tokens, power_w, model_size, model_name
        )
        energy_results = None
        if 'energy_per_token' in df_sorted.columns:
            energy_per_token = df_sorted['energy_per_token'].values
            energy_results = self.energy_fitter.fit_energy_trend(
                output_tokens, energy_per_token, model_size, model_name
            )
        
        sys5v_data = None
        if 'avg_power_w_sys5v' in df_sorted.columns:
            sys5v_power = df_sorted['avg_power_w_sys5v'].values 
            valid_mask = sys5v_power > 0
            if valid_mask.any():
                sys5v_data = {
                    'x_data': output_tokens[valid_mask],
                    'y_data': sys5v_power[valid_mask]
                }
        
        results = {
            'model_name': model_name,
            'model_size': model_size,
            'power_fit': power_results,
            'energy_fit': energy_results,
            'data_points': len(output_tokens)
        }
        
        if sys5v_data:
            results['sys5v_data'] = sys5v_data
            
        self.fitted_models[model_name] = results
        return results
    
    def fit_all_models(self, correlation_file: str, input_length_filter: int = None) -> Dict[str, Dict]:
        """
        Fit all models from the correlation file.
        
        Args:
            correlation_file: Path to correlation Excel file
            input_length_filter: If specified, only use data with this input length
        """
        model_data = self.load_correlation_data(correlation_file)
        results = {}
        
        for model_name, df in model_data.items():
            try:
                # Apply input length filter if specified
                if input_length_filter and 'input_tokens' in df.columns:
                    original_count = len(df)
                    df = df[df['input_tokens'] == input_length_filter].copy()
                    filtered_count = len(df)
                    print(f"  {model_name}: Using {filtered_count} records (filtered from {original_count}) for input={input_length_filter}")
                    
                    if df.empty:
                        print(f"  ⚠️ No data found for {model_name} with input length {input_length_filter}")
                        continue
                
                model_results = self.fit_model_data(df, model_name)
                if model_results:
                    # Store input length info in results
                    model_results['input_length_filter'] = input_length_filter
                    results[model_name] = model_results
            except Exception as e:
                print(f"Error fitting {model_name}: {e}")
                continue
        return results
    
    def generate_comparison_plots(self, results: Dict[str, Dict], output_dir: str):
        """Generate comparison plots across all models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        def format_model_name(model_name):
            return model_name
        def get_model_sort_key(model_name):
            if '14B' in model_name:
                return 0
            elif '8B' in model_name:
                return 1
            elif '1.5B' in model_name:
                return 2
            else:
                return 999
        sorted_results = sorted(results.items(), key=lambda x: get_model_sort_key(x[0]))
        
        all_x_data = []
        for model_name, model_results in sorted_results:
            power_fit = model_results.get('power_fit', {})
            if 'x_data' in power_fit:
                all_x_data.extend(power_fit['x_data'])
        
        if all_x_data:
            global_x_min = 1
            global_x_max = max(max(all_x_data), 2048)
        else:
            global_x_min, global_x_max = 1, 2048
        
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT*1.5))
        
        for i, (model_name, model_results) in enumerate(sorted_results):
            color = colors[i % len(colors)]
            power_fit = model_results.get('power_fit', {})
            if 'x_data' in power_fit and 'y_pred' in power_fit:
                x_data = power_fit['x_data']
                y_data = power_fit['y_data']
                r2_score = power_fit['r2_score']
                display_name = format_model_name(model_name)
                
                ax1.scatter(x_data, y_data, alpha=0.7, color=color, s=50, label=display_name)
                
                if 'x_smooth' in power_fit and 'y_smooth' in power_fit:
                    if power_fit['x_smooth'][0] > global_x_min or power_fit['x_smooth'][-1] < global_x_max:
                        if 'fitted_params' in power_fit and 'function' in power_fit:
                            x_smooth = np.linspace(global_x_min, global_x_max, 200)
                            y_smooth = power_fit['function'](x_smooth, *power_fit['fitted_params'])
                        else:
                            x_smooth = power_fit['x_smooth']
                            y_smooth = power_fit['y_smooth']
                    else:
                        x_smooth = power_fit['x_smooth']
                        y_smooth = power_fit['y_smooth']
                else:
                    sort_idx = np.argsort(x_data)
                    x_smooth = x_data[sort_idx]
                    y_smooth = power_fit['y_pred'][sort_idx]
                
                ax1.plot(x_smooth, y_smooth, '--', 
                        color=color, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Output Length', fontsize=FONT_SIZE_LABELS)
        ax1.set_ylabel('Power (W)', fontsize=FONT_SIZE_LABELS)
        ax1.legend(loc='center right', fontsize=FONT_SIZE_LEGEND, bbox_to_anchor=(0.98, 0.5))
        ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        # Add input length to title if filtered
        input_filter = None
        for model_results in results.values():
            if 'input_length_filter' in model_results and model_results['input_length_filter']:
                input_filter = model_results['input_length_filter']
                break
        if input_filter:
            ax1.set_title(f'Power Scaling (Input: {input_filter} tokens)', fontsize=FONT_SIZE_LABELS)
        
        # Plot 2: SYS5V Power (scatter only, no fits)
        has_sys5v_data = False
        for i, (model_name, model_results) in enumerate(sorted_results):
            color = colors[i % len(colors)]
            display_name = format_model_name(model_name)
            
            if 'sys5v_data' in model_results:
                has_sys5v_data = True
                sys5v_x = model_results['sys5v_data']['x_data']
                sys5v_y = model_results['sys5v_data']['y_data']
                ax2.scatter(sys5v_x, sys5v_y, alpha=0.7, color=color, s=50, 
                          marker='o', label=display_name)
        
        if has_sys5v_data:
            ax2.set_xlabel('Output Length', fontsize=FONT_SIZE_LABELS)
            ax2.set_ylabel('SYS5V Power (W)', fontsize=FONT_SIZE_LABELS)
            ax2.legend(loc='center right', fontsize=FONT_SIZE_LEGEND, bbox_to_anchor=(0.98, 0.5))
            ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)

            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        else:
            # Hide the second subplot if no SYS5V data
            ax2.set_visible(False)
            ax1.set_xlabel('Output Length', fontsize=FONT_SIZE_LABELS)
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        power_pdf_path = output_path / 'power_consumption_fits.pdf'
        plt.savefig(power_pdf_path, bbox_inches='tight')
        plt.close()
        print(f"Saved power consumption plot to: {power_pdf_path}")
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        has_energy_data = False
        for i, (model_name, model_results) in enumerate(sorted_results):
            color = colors[i % len(colors)]
            energy_fit = model_results.get('energy_fit', {})
            if energy_fit and 'x_data' in energy_fit and 'y_pred' in energy_fit:
                has_energy_data = True
                x_data = energy_fit['x_data']
                y_data = energy_fit['y_data']
                r2_score = energy_fit['r2_score']
                display_name = format_model_name(model_name)
                
                # Plot data points
                ax2.scatter(x_data, y_data, alpha=0.7, color=color, s=50)
                
                if 'x_smooth' in energy_fit and 'y_smooth' in energy_fit:
                    if energy_fit['x_smooth'][0] > global_x_min or energy_fit['x_smooth'][-1] < global_x_max:
                        if 'fitted_params' in energy_fit and 'function' in energy_fit:
                            x_smooth = np.linspace(global_x_min, global_x_max, 200)
                            y_smooth = energy_fit['function'](x_smooth, *energy_fit['fitted_params'])
                        else:
                            x_smooth = energy_fit['x_smooth']
                            y_smooth = energy_fit['y_smooth']
                    else:
                        x_smooth = energy_fit['x_smooth']
                        y_smooth = energy_fit['y_smooth']
                else:
                    sort_idx = np.argsort(x_data)
                    x_smooth = x_data[sort_idx]
                    y_smooth = energy_fit['y_pred'][sort_idx]
                
                if SHOW_R2_IN_LEGEND:
                    legend_label = f'{display_name} (R² = {r2_score:.3f})'
                else:
                    legend_label = display_name
                ax2.plot(x_smooth, y_smooth, '--', 
                        color=color, linewidth=2, label=legend_label)
        
        if has_energy_data:
            ax2.set_xlabel('Output Length', fontsize=FONT_SIZE_LABELS)
            ax2.set_ylabel('Energy (J/token)', fontsize=FONT_SIZE_LABELS)
            ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left', bbox_to_anchor=(0.02, 1.02))
            ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
            
            # Add input length to title if filtered
            if input_filter:
                ax2.set_title(f'Energy per Token (Input: {input_filter} tokens)', fontsize=FONT_SIZE_LABELS)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            energy_pdf_path = output_path / 'energy_j_token_fits.pdf'
            plt.savefig(energy_pdf_path, bbox_inches='tight')
            plt.close()
            print(f"Saved energy (J/token) plot to: {energy_pdf_path}")
        else:
            plt.close()
            print("No energy per token data available for plotting")

    def save_fitting_summary(self, results: Dict[str, Dict], output_file: str):
        """Save fitting results to JSON file."""
        summary = []
        for model_name, model_results in results.items():
            power_fit = model_results.get('power_fit', {})
            energy_fit = model_results.get('energy_fit', {})
            model_summary = {
                'Model': model_name,
                'Model_Size': model_results['model_size'],
                'Data_Points': model_results['data_points'],
                'Power_Function': power_fit.get('function_name', 'N/A'),
                'Power_R2': power_fit.get('r2_score', 0.0),
                'Power_Parameters': power_fit.get('parameters', {}),
            }
            if energy_fit:
                model_summary.update({
                    'Energy_Function': energy_fit.get('function_name', 'N/A'),
                    'Energy_R2': energy_fit.get('r2_score', 0.0),
                    'Energy_Parameters': energy_fit.get('parameters', {}),
                    'Energy_Unit': 'joules' 
                })
            summary.append(model_summary)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved fitting summary to: {output_path}")

    def plot_individual_fits(self, results: Dict[str, Dict], output_dir: str):
        """Generate individual plots for each model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for model_name, model_results in results.items():
            power_fit = model_results.get('power_fit', {})
            if power_fit:
                self.power_fitter.plot_power_fit(power_fit, model_name, output_path)
            energy_fit = model_results.get('energy_fit', {})
            if energy_fit:
                self.energy_fitter.plot_energy_fit(energy_fit, model_name, output_path) 
