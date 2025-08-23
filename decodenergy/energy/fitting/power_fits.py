"""
Power fitting functions for different model sizes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

FIGURE_WIDTH = 4
FIGURE_HEIGHT = 4
FONT_SIZE_LABELS = 12
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICKS = 12
SHOW_R2_IN_LEGEND = False

class PowerFitter:
    """Handles power consumption fitting for different model sizes using logarithmic functions."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def constant_function(x, c):
        """Constant function: y = c"""
        return np.full_like(x, c)
    
    @staticmethod
    def linear_function(x, a, b):
        """Linear function: y = a*x + b"""
        return a * x + b

    @staticmethod
    def log_function(x, a, b):
        """Logarithmic function: y = a*ln(x) + b"""
        return a * np.log(x) + b

    def fit_1_5b_power(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 1.5B model power: logarithmic fit."""
        try:
            popt, pcov = curve_fit(self.log_function, x_data, y_data)
            y_pred = self.log_function(x_data, *popt)
            
            x_min = 1
            x_max = max(x_data.max(), 2048)
            x_smooth = np.linspace(x_min, x_max, 200)
            y_smooth = self.log_function(x_smooth, *popt)
            
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': '1.5B_logarithmic',
                'parameters': {
                    'scale': popt[0],
                    'offset': popt[1]
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'function': self.log_function,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 1.5B power (logarithmic): {e}")
            raise RuntimeError(f"Failed to fit 1.5B power with logarithmic function: {e}")

    def fit_8b_power(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 8B model power: logarithmic fit."""
        try:
            popt, pcov = curve_fit(self.log_function, x_data, y_data)
            y_pred = self.log_function(x_data, *popt)
            
            x_min = 1
            x_max = max(x_data.max(), 2048)
            x_smooth = np.linspace(x_min, x_max, 200)
            y_smooth = self.log_function(x_smooth, *popt)
            
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': '8B_logarithmic',
                'parameters': {
                    'scale': popt[0],
                    'offset': popt[1]
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'function': self.log_function,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 8B power (logarithmic): {e}")
            raise RuntimeError(f"Failed to fit 8B power with logarithmic function: {e}")

    def fit_14b_power(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 14B model power: logarithmic fit."""
        try:
            popt, pcov = curve_fit(self.log_function, x_data, y_data)
            y_pred = self.log_function(x_data, *popt)
            
            x_min = 1
            x_max = max(x_data.max(), 2048)
            x_smooth = np.linspace(x_min, x_max, 200)
            y_smooth = self.log_function(x_smooth, *popt)
            
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return {
                'function_name': '14B_logarithmic',
                'parameters': {
                    'scale': popt[0],
                    'offset': popt[1]
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'function': self.log_function,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 14B power (logarithmic): {e}")
            raise RuntimeError(f"Failed to fit 14B power with logarithmic function: {e}")

    def fit_power_trend(self, x_data: np.ndarray, y_data: np.ndarray, 
                       model_size: str, model_name: str) -> Dict:
        """
        Fit power trend based on model size.
        
        Args:
            x_data: Decode token counts
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
            print(f"  Unknown model size {model_size}, no logarithmic fit available")
            raise RuntimeError(f"No logarithmic fitting function available for model size: {model_size}")

    def plot_power_fit(self, fit_results: Dict, model_name: str, output_dir: Path):
        """Plot individual power fit results with smooth logarithmic curves."""
        if 'x_data' not in fit_results:
            return
            
        x_data = fit_results['x_data']
        y_data = fit_results['y_data']
        x_smooth = fit_results.get('x_smooth', x_data)
        y_smooth = fit_results.get('y_smooth', fit_results.get('y_pred', y_data))
        
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        display_name = f'DSR1-{model_name}' if model_name in ['8B', '1.5B', '14B'] else model_name
        
        # Plot data points
        plt.scatter(x_data, y_data, alpha=0.7, label='Actual', color='#1f77b4')
        
        # Plot smooth logarithmic curve
        if SHOW_R2_IN_LEGEND:
            fitted_label = f'Fitted (R² = {fit_results["r2_score"]:.3f})'
        else:
            fitted_label = 'Fitted'
        
        plt.plot(x_smooth, y_smooth, '--', linewidth=2, color='#ff7f0e',
                label=fitted_label)
        
        plt.xlabel('Output Length', fontsize=FONT_SIZE_LABELS)
        plt.ylabel('Average Power (W)', fontsize=FONT_SIZE_LABELS)
        plt.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right', bbox_to_anchor=(0.98, 0.02))
        plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        # Set x-axis ticks to multiples of 128, excluding 1024 for space
        import numpy as np
        x_min, x_max = plt.xlim()
        tick_start = int(np.ceil(x_min / 128) * 128)
        tick_end = int(np.floor(x_max / 128) * 128) + 1
        ticks = np.arange(tick_start, tick_end, 128)
        plt.xticks(ticks)
        
        pdf_path = output_dir / f'{model_name.lower()}_power_fit.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved individual plot: {pdf_path}") 
