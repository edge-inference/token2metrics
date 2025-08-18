"""
Power fitting functions for different model sizes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import pwlf

FIGURE_WIDTH = 8
FIGURE_HEIGHT = 6
FONT_SIZE_LABELS = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICKS = 12
SHOW_R2_IN_LEGEND = False

class PowerFitter:
    """
    Fits power consumption trends for different model sizes using segmented regression.
    Linear segment → Logarithmic segment at specified transition points.
    """
    def __init__(self):
        self.transition_points = {
            '1.5B': 3000,
            '8B': 800,
            '14B': 800
        }
    
    @staticmethod
    def constant_function(x, c):
        """Constant function: y = c"""
        return np.full_like(x, c)
    
    @staticmethod
    def log_function(x, a, b):
        """Logarithmic function: y = a*ln(x) + b"""
        return a * np.log(x) + b

    @staticmethod
    def linear_function(x, a, b):
        """Linear function: y = ax + b (for fallback only)"""
        return a * x + b
    
    def fit_segmented_regression(self, x_data: np.ndarray, y_data: np.ndarray, 
                                transition: int, model_name: str) -> Dict:
        """
        Fit segmented regression: constant → logarithmic at transition point.
        
        Args:
            x_data: Input token counts
            y_data: Power consumption values
            transition: Transition point between constant and log segments
            model_name: Model name for identification
            
        Returns:
            Dictionary with fitting results
        """
        try:
            constant_mask = x_data <= transition
            log_mask = x_data > transition
            
            if not np.any(constant_mask) or not np.any(log_mask):
                return self._fallback_linear_fit(x_data, y_data, f'{model_name}_fallback_linear')
            
            y_constant = y_data[constant_mask]
            constant_value = np.mean(y_constant)
            
            x_log = x_data[log_mask]
            y_log = y_data[log_mask]
            log_params, _ = curve_fit(self.log_function, x_log, y_log)
            
            y_pred = np.zeros_like(x_data, dtype=float)
            y_pred[constant_mask] = constant_value
            y_pred[log_mask] = self.log_function(x_log, *log_params)
            
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            return {
                'function_name': f'{model_name}_constant_to_{transition}_then_log',
                'parameters': {
                    'constant_value': constant_value,
                    'log_scale': log_params[0],
                    'log_offset': log_params[1],
                    'transition_point': transition
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'constant_value': constant_value,
                'log_params': log_params,
                'transition': transition
            }
            
        except Exception as e:
            print(f"Error fitting {model_name} power: {e}")
            return self._fallback_linear_fit(x_data, y_data, f'{model_name}_fallback_linear')

    def fit_1_5b_power(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 1.5B model power: linear till 3000, then log."""
        transition = self.transition_points['1.5B']
        return self.fit_segmented_regression(x_data, y_data, transition, '1.5B')

    def fit_8b_power(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 8B model power: linear till 800, then log."""
        transition = self.transition_points['8B']
        return self.fit_segmented_regression(x_data, y_data, transition, '8B')

    def fit_14b_power(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 14B model power: linear till 800, then log."""
        transition = self.transition_points['14B']
        return self.fit_segmented_regression(x_data, y_data, transition, '14B')

    def _fallback_linear_fit(self, x_data: np.ndarray, y_data: np.ndarray, function_name: str) -> Dict:
        """Fallback to simple linear fit if segmented fitting fails."""
        try:
            popt, pcov = curve_fit(self.linear_function, x_data, y_data)
            y_pred = self.linear_function(x_data, *popt)
            
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
                'function': self.linear_function,
                'fitted_params': popt
            }
        except:
            return {'function_name': 'failed', 'r2_score': 0.0}

    def fit_power_trend(self, x_data: np.ndarray, y_data: np.ndarray, 
                       model_size: str, model_name: str) -> Dict:
        """
        Fit power trend based on model size.
        
        Args:
            x_data: Input token counts
            y_data: Power consumption values
            model_size: Detected model size ('1.5B', '8B', '14B', etc.)
            model_name: Full model name for logging
            
        Returns:
            Dictionary with fitting results
        """
        print(f"  Fitting power trend for {model_name} ({model_size})")
        
        if model_size == '1.5B':
            return self.fit_1_5b_power(x_data, y_data)
        elif model_size == '8B':
            return self.fit_8b_power(x_data, y_data)
        elif model_size == '14B':
            return self.fit_14b_power(x_data, y_data)
        else:
            print(f"  Unknown model size {model_size}, using linear fallback")
            return self._fallback_linear_fit(x_data, y_data, f'{model_size}_linear_fallback')

    def plot_power_fit(self, fit_results: Dict, model_name: str, output_dir: Path):
        """Plot individual power fit results."""
        if 'x_data' not in fit_results:
            return
            
        x_data = fit_results['x_data']
        y_data = fit_results['y_data']
        y_pred = fit_results['y_pred']
        
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        display_name = f'DSR1-{model_name}' if model_name in ['8B', '1.5B', '14B'] else model_name
        
        plt.scatter(x_data, y_data, alpha=0.7, label='Actual', color='#1f77b4')
        
        sort_idx = np.argsort(x_data)
        
        if SHOW_R2_IN_LEGEND:
            fitted_label = f'Fitted (R² = {fit_results["r2_score"]:.3f})'
        else:
            fitted_label = 'Fitted'
        
        plt.plot(x_data[sort_idx], y_pred[sort_idx], '--', linewidth=2, color='#ff7f0e',
                label=fitted_label)
        
        plt.xlabel('Prefill Tokens', fontsize=FONT_SIZE_LABELS)
        plt.ylabel('Average Power (W)', fontsize=FONT_SIZE_LABELS)
        plt.legend(fontsize=FONT_SIZE_LEGEND)
        plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        png_path = output_dir / f'{model_name.lower()}_power_fit.png'
        pdf_path = output_dir / f'{model_name.lower()}_power_fit.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved individual plot: {png_path}")
        print(f"    Saved individual plot: {pdf_path}") 
