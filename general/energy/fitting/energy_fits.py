"""
Energy per token fitting functions for different model sizes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

FIGURE_WIDTH = 8
FIGURE_HEIGHT = 6
FONT_SIZE_LABELS = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICKS = 12
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
    def exponential_decay(x, a, b, c):
        """Exponential decay: y = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c
    
    @staticmethod
    def polynomial_2(x, a, b, c):
        """Quadratic polynomial: y = ax² + bx + c"""
        return a * x**2 + b * x + c
    
    @staticmethod
    def polynomial_3(x, a, b, c, d):
        """Cubic polynomial: y = ax³ + bx² + cx + d"""
        return a * x**3 + b * x**2 + c * x + d
    
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
        """Fit 8B model energy per token: linear behavior."""
        try:
            popt, pcov = curve_fit(self.linear_function, x_data, y_data)
            y_pred = self.linear_function(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': '8B_linear',
                'parameters': {
                    'slope': popt[0],
                    'intercept': popt[1]
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'function': self.linear_function,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 8B energy (linear): {e}")
            return self._fallback_polynomial_fit(x_data, y_data, '8B_fallback_poly')
    
    def fit_1_5b_energy(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 1.5B model energy per token: try multiple functions and pick best."""
        functions_to_try = [
            (self.polynomial_2, [0.001, -0.1, 1], '1.5B_quadratic'),
            (self.polynomial_3, [0.0001, -0.001, 0.1, 1], '1.5B_cubic'),
            (self.power_law, [1, -0.5, 0.1], '1.5B_power_law'),
            (self.exponential_decay, [1, 0.001, 0.1], '1.5B_exponential_decay')
        ]
        best_fit = None
        best_r2 = -np.inf
        for func, p0, name in functions_to_try:
            try:
                popt, pcov = curve_fit(func, x_data, y_data, p0=p0, maxfev=5000)
                y_pred = func(x_data, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                if r2 > best_r2:
                    best_r2 = r2
                    if func == self.polynomial_2:
                        params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
                    elif func == self.polynomial_3:
                        params = {'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3]}
                    elif func == self.power_law:
                        params = {'amplitude': popt[0], 'exponent': popt[1], 'offset': popt[2]}
                    elif func == self.exponential_decay:
                        params = {'amplitude': popt[0], 'decay_rate': popt[1], 'baseline': popt[2]}
                    else:
                        params = {f'param_{i}': p for i, p in enumerate(popt)}
                    best_fit = {
                        'function_name': name,
                        'parameters': params,
                        'r2_score': r2,
                        'x_data': x_data,
                        'y_data': y_data,
                        'y_pred': y_pred,
                        'function': func,
                        'fitted_params': popt
                    }
            except Exception as e:
                print(f"  Failed to fit {name}: {e}")
                continue
        if best_fit is None:
            print("All fitting attempts failed for 1.5B energy, using polynomial fallback")
            return self._fallback_polynomial_fit(x_data, y_data, '1.5B_fallback_poly')
        print(f"  Best fit for 1.5B energy: {best_fit['function_name']} (R² = {best_r2:.4f})")
        return best_fit
    
    def _fallback_polynomial_fit(self, x_data: np.ndarray, y_data: np.ndarray, function_name: str) -> Dict:
        """Fallback to quadratic polynomial fit."""
        try:
            popt, pcov = curve_fit(self.polynomial_2, x_data, y_data)
            y_pred = self.polynomial_2(x_data, *popt)
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
                'function': self.polynomial_2,
                'fitted_params': popt
            }
        except:
            return {'function_name': 'failed', 'r2_score': 0.0}
    
    def fit_energy_trend(self, x_data: np.ndarray, y_data: np.ndarray, 
                        model_size: str, model_name: str) -> Dict:
        """
        Fit energy per token trend based on model size.
        
        Args:
            x_data: Input token counts
            y_data: Energy per token values
            model_size: Detected model size ('1.5B', '8B', '14B', etc.)
            model_name: Full model name for logging
            
        Returns:
            Dictionary with fitting results
        """
        print(f"  Fitting energy per token trend for {model_name} ({model_size})")
        if model_size == '14B':
            return self.fit_14b_energy(x_data, y_data)
        elif model_size == '8B':
            return self.fit_8b_energy(x_data, y_data)
        elif model_size == '1.5B':
            return self.fit_1_5b_energy(x_data, y_data)
        else:
            print(f"  Unknown model size {model_size}, using polynomial fallback")
            return self._fallback_polynomial_fit(x_data, y_data, f'{model_size}_poly_fallback')
    
    def plot_energy_fit(self, fit_results: Dict, model_name: str, output_dir: Path):
        """Plot individual energy per token fit results."""
        if 'x_data' not in fit_results:
            return
        x_data = fit_results['x_data']
        y_data = fit_results['y_data']
        y_pred = fit_results['y_pred']
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        display_name = f'DSR1-{model_name}' if model_name in ['8B', '1.5B', '14B'] else model_name
        ax.scatter(x_data, y_data, alpha=0.7, color='#1f77b4', label='Actual')
        if SHOW_R2_IN_LEGEND:
            fitted_label = f'Fitted (R² = {fit_results["r2_score"]:.4f})'
        else:
            fitted_label = 'Fitted'
        ax.plot(x_data, y_pred, '--', linewidth=2, color='#ff7f0e', label=fitted_label)
        ax.set_xlabel('Prefill Tokens', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy (J/token)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        plt.tight_layout()
        safe_name = model_name.replace('/', '_').replace('\\', '_')
        png_path = output_dir / f'{safe_name}_energy_fit.png'
        pdf_path = output_dir / f'{safe_name}_energy_fit.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"    Saved energy fit plot to: {png_path}")
        print(f"    Saved energy fit plot to: {pdf_path}")
        plt.close()
