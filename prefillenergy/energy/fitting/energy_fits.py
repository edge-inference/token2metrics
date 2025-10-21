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
Energy per token fitting functions for different model sizes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

FIGURE_WIDTH = 6
FIGURE_HEIGHT = 5
FONT_SIZE_LABELS = 16
FONT_SIZE_LEGEND = 14
FONT_SIZE_ENERGY_LEGEND = 10  
FONT_SIZE_TICKS = 14
SHOW_R2_IN_LEGEND = False

class EnergyFitter:
    """Handles energy per token fitting for different model sizes."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def linear_function(x, a, b):
        """Linear function: y = ax + b"""
        return a * x + b
    
    @staticmethod
    def log_function(x, a, b):
        """Logarithmic function: y = a*ln(x) + b"""
        return a * np.log(x) + b
    
    @staticmethod
    def exponential_decay(x, a, b, c):
        """Exponential decay: y = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c
    
    @staticmethod
    def polynomial_2(x, a, b, c):
        """Quadratic polynomial: y = ax² + bx + c"""
        return a * x**2 + b * x + c
    
    @staticmethod
    def power_law(x, a, b, c):
        """Power law: y = a * x^b + c"""
        return a * np.power(x, b) + c
    
    def fit_14b_energy(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 14B model energy per token: exponential decay ~ e^-x."""
        try:
            y_max = np.max(y_data)
            y_min = np.min(y_data)
            p0 = [y_max - y_min, 0.001, y_min]
            popt, pcov = curve_fit(self.exponential_decay, x_data, y_data, p0=p0, maxfev=5000)
            y_pred = self.exponential_decay(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': '14B_exponential_decay',
                'parameters': {
                    'amplitude': popt[0],
                    'decay_rate': popt[1],
                    'baseline': popt[2]
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'function': self.exponential_decay,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 14B energy (exponential): {e}")
            return self._fallback_polynomial_fit(x_data, y_data, '14B_fallback_poly')
    
    def fit_8b_energy(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 8B model energy per token: exponential decay behavior."""
        try:
            # Use exponential decay similar to 14B model
            y_max = np.max(y_data)
            y_min = np.min(y_data)
            # Initial parameter estimates for exponential decay
            p0 = [y_max - y_min, 0.001, y_min]
            popt, pcov = curve_fit(self.exponential_decay, x_data, y_data, p0=p0, maxfev=5000)
            y_pred = self.exponential_decay(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': '8B_exponential_decay',
                'parameters': {
                    'amplitude': popt[0],
                    'decay_rate': popt[1],
                    'baseline': popt[2]
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'function': self.exponential_decay,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 8B energy (exponential decay): {e}")
            print("Falling back to linear fit for 8B energy")
            return self._fallback_linear_fit(x_data, y_data, '8B_fallback_linear')
    
    def fit_1_5b_energy(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 1.5B model energy per token: exponential decay behavior."""
        try:
            y_max = np.max(y_data)
            y_min = np.min(y_data)
            p0 = [y_max - y_min, 0.001, y_min]
            popt, pcov = curve_fit(self.exponential_decay, x_data, y_data, p0=p0, maxfev=5000)
            y_pred = self.exponential_decay(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': '1.5B_exponential_decay',
                'parameters': {
                    'amplitude': popt[0],
                    'decay_rate': popt[1],
                    'baseline': popt[2]
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'function': self.exponential_decay,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 1.5B energy (exponential decay): {e}")
            print("Falling back to polynomial fit for 1.5B energy")
            return self._fallback_polynomial_fit(x_data, y_data, '1.5B_fallback_poly')
    
    def _fallback_polynomial_fit(self, x_data: np.ndarray, y_data: np.ndarray, function_name: str) -> Dict:
        """Fallback to quadratic polynomial fit with smooth interpolation."""
        try:
            popt, pcov = curve_fit(self.polynomial_2, x_data, y_data)
            y_pred = self.polynomial_2(x_data, *popt)
            
            x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            y_pred_smooth = self.polynomial_2(x_smooth, *popt)
            
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': function_name,
                'parameters': {'a': popt[0], 'b': popt[1], 'c': popt[2]},
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_pred_smooth': y_pred_smooth,
                'function': self.polynomial_2,
                'fitted_params': popt
            }
        except:
            return {'function_name': 'failed', 'r2_score': 0.0}
    
    def _fallback_linear_fit(self, x_data: np.ndarray, y_data: np.ndarray, function_name: str) -> Dict:
        """Fallback to simple linear fit with smooth interpolation."""
        try:
            popt, pcov = curve_fit(self.linear_function, x_data, y_data)
            y_pred = self.linear_function(x_data, *popt)
            
            x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            y_pred_smooth = self.linear_function(x_smooth, *popt)
            
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': function_name,
                'parameters': {'slope': popt[0], 'intercept': popt[1]},
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_pred_smooth': y_pred_smooth,
                'function': self.linear_function,
                'fitted_params': popt
            }
        except:
            return {'function_name': 'failed', 'r2_score': 0.0}
    
    def fit_energy_trend(self, x_data: np.ndarray, y_data: np.ndarray, 
                        model_size: str, model_name: str) -> Dict:
        """
        Tries multiple functions and picks the best one based on R².
        
        Args:
            x_data: Input token counts
            y_data: Energy per token values
            model_size: Detected model size ('1.5B', '8B', '14B', etc.)
            model_name: Full model name for logging
            
        Returns:
            Dictionary with fitting results
        """
        print(f"  Fitting energy per token trend for {model_name} ({model_size}) using adaptive approach")
        
        fitting_approaches = []
        
        for degree in [1, 2]:
            if len(x_data) > degree:
                try:
                    coeffs = np.polyfit(x_data, y_data, degree)
                    poly_func = np.poly1d(coeffs)
                    y_pred = poly_func(x_data)
                    r2 = self._calculate_r2(y_data, y_pred)
                    
                    x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
                    y_pred_smooth = poly_func(x_smooth)
                    
                    fitting_approaches.append({
                        'name': f'polynomial_deg{degree}',
                        'r2': r2,
                        'function': poly_func,
                        'params': coeffs.tolist(),
                        'x_smooth': x_smooth,
                        'y_pred_smooth': y_pred_smooth,
                        'y_pred': y_pred
                    })
                except:
                    continue
        
        # 2. Try logarithmic fitting
        if np.all(x_data > 0):
            try:
                popt, _ = curve_fit(self.log_function, x_data, y_data)
                y_pred = self.log_function(x_data, *popt)
                r2 = self._calculate_r2(y_data, y_pred)
                
                # Generate smooth curve
                x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
                y_pred_smooth = self.log_function(x_smooth, *popt)
                
                fitting_approaches.append({
                    'name': 'logarithmic',
                    'r2': r2,
                    'function': self.log_function,
                    'params': popt.tolist(),
                    'x_smooth': x_smooth,
                    'y_pred_smooth': y_pred_smooth,
                    'y_pred': y_pred
                })
            except:
                pass
        
        # 3. Try exponential decay
        try:
            y_max = np.max(y_data)
            y_min = np.min(y_data)
            p0 = [y_max - y_min, 0.001, y_min]
            popt, _ = curve_fit(self.exponential_decay, x_data, y_data, p0=p0, maxfev=5000)
            y_pred = self.exponential_decay(x_data, *popt)
            r2 = self._calculate_r2(y_data, y_pred)
            
            # Generate smooth curve
            x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            y_pred_smooth = self.exponential_decay(x_smooth, *popt)
            
            fitting_approaches.append({
                'name': 'exponential_decay',
                'r2': r2,
                'function': self.exponential_decay,
                'params': popt.tolist(),
                'x_smooth': x_smooth,
                'y_pred_smooth': y_pred_smooth,
                'y_pred': y_pred
            })
        except:
            pass
        
        if not fitting_approaches:
            print(f"  All fitting approaches failed for {model_name}, using linear fallback")
            return self._fallback_linear_fit(x_data, y_data, f'{model_name}_linear_fallback')
        
        best_fit = max(fitting_approaches, key=lambda x: x['r2'])
        print(f"  Best fit for {model_name}: {best_fit['name']} (R² = {best_fit['r2']:.4f})")
        
        return {
            'function_name': f'{model_name}_{best_fit["name"]}',
            'parameters': {
                'function_type': best_fit['name'],
                'coefficients': best_fit['params']
            },
            'r2_score': best_fit['r2'],
            'x_data': x_data,
            'y_data': y_data,
            'y_pred': best_fit['y_pred'],
            'x_smooth': best_fit['x_smooth'],
            'y_pred_smooth': best_fit['y_pred_smooth'],
            'function': best_fit['function'],
            'fitted_params': best_fit['params']
        }
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_energy_fit(self, fit_results: Dict, model_name: str, output_dir: Path):
        """Plot individual energy per token fit results."""
        if 'x_data' not in fit_results:
            return
        x_data = fit_results['x_data']
        y_data = fit_results['y_data']
        
        if 'x_smooth' in fit_results and 'y_pred_smooth' in fit_results:
            x_plot = fit_results['x_smooth']
            y_plot = fit_results['y_pred_smooth']
        else:
            x_plot = x_data
            y_plot = fit_results['y_pred']
        
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        display_name = f'DSR1-{model_name}' if model_name in ['8B', '1.5B', '14B'] else model_name
        ax.scatter(x_data, y_data, alpha=0.7, color='#1f77b4', label='Actual')
        if SHOW_R2_IN_LEGEND:
            fitted_label = f'Fitted (R² = {fit_results["r2_score"]:.4f})'
        else:
            fitted_label = 'Fitted'
        ax.plot(x_plot, y_plot, '--', linewidth=2, color='#ff7f0e', label=fitted_label)
        ax.set_xlabel('Input Length', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy (J/token)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_ENERGY_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        current_xlim = ax.get_xlim()
        ax.set_xticks(range(0, min(int(current_xlim[1]) + 1000, 4500), 500))
        plt.tight_layout()
        safe_name = model_name.replace('/', '_').replace('\\', '_')
        png_path = output_dir / f'{safe_name}_energy_fit.png'
        pdf_path = output_dir / f'{safe_name}_energy_fit.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"    Saved energy fit plot to: {png_path}")
        print(f"    Saved energy fit plot to: {pdf_path}")
        plt.close()

    def generate_energy_lookup_table(self, fitted_models: Dict, token_counts: List[int] = None) -> Dict:
        """
        Generate a JSON lookup table for energy per token values.
        
        Args:
            fitted_models: Dictionary containing fitted model results
            token_counts: List of token counts to generate values for
            
        Returns:
            Dictionary with model lookup tables
        """
        if token_counts is None:
            # Default token counts for lookup table
            token_counts = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 4096]
        
        lookup_table = {
            "metadata": {
                "description": "Energy per token lookup table",
                "token_counts": token_counts,
                "models": list(fitted_models.keys()),
                "function_type": "exponential_decay"
            },
            "models": {}
        }
        
        for model_name, model_results in fitted_models.items():
            if 'energy_fit' not in model_results:
                continue
                
            energy_fit = model_results['energy_fit']
            if 'function' not in energy_fit or 'fitted_params' not in energy_fit:
                continue
            
            # Generate energy values for each token count
            model_data = {}
            func = energy_fit['function']
            params = energy_fit['fitted_params']
            
            for token_count in token_counts:
                energy_value = func(token_count, *params)
                # Round to reasonable precision (4 decimal places)
                model_data[str(token_count)] = round(float(energy_value), 4)
            
            # Add model parameters for reference
            model_info = {
                "energy_per_token": model_data,
                "fit_parameters": {
                    "amplitude": round(float(params[0]), 6),
                    "decay_rate": round(float(params[1]), 6), 
                    "baseline": round(float(params[2]), 6)
                }
            }
            
    def generate_lookup_table(self, results: Dict, input_lengths: List[int] = None) -> Dict:
        """
        Generate a lookup table for energy per token values without units.
        
        Args:
            results: Dictionary of fitted results for each model
            input_lengths: List of input token lengths to calculate values for
            
        Returns:
            Dictionary with lookup table structure
        """
        if input_lengths is None:
            input_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        
        lookup_table = {
            "description": "Energy per token lookup table (values in J/token, no units shown)",
            "input_lengths": input_lengths,
            "models": {}
        }
        
        for model_name, model_results in results.items():
            energy_fit = model_results.get('energy_fit', {})
            if 'function' not in energy_fit or 'fitted_params' not in energy_fit:
                continue
                
            func = energy_fit['function']
            params = energy_fit['fitted_params']
            
            # Calculate energy values for each input length
            energy_values = {}
            for length in input_lengths:
                try:
                    value = func(length, *params)
                    energy_values[str(length)] = round(float(value), 6)
                except:
                    energy_values[str(length)] = None
            
            model_info = {
                "energy_lookup": energy_values,
                "function_type": energy_fit.get('function_name', 'unknown'),
                "fitted_parameters": {
                    "amplitude": round(float(params[0]), 6),
                    "decay_rate": round(float(params[1]), 6), 
                    "baseline": round(float(params[2]), 6)
                },
                "r2_score": round(energy_fit.get('r2_score', 0.0), 4)
            }
            
            lookup_table["models"][model_name] = model_info
        
        return lookup_table
