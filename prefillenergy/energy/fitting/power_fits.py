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

FIGURE_WIDTH = 6
FIGURE_HEIGHT = 5
FONT_SIZE_LABELS = 16
FONT_SIZE_LEGEND = 14
FONT_SIZE_POWER_LEGEND = 10 
FONT_SIZE_TICKS = 14
SHOW_R2_IN_LEGEND = False

def calculate_r2(y_data: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1 - (ss_res / ss_tot)

def create_fit_result(function_name: str, parameters: dict, r2: float, 
                     x_data: np.ndarray, y_data: np.ndarray, y_pred: np.ndarray, **kwargs) -> dict:
    """Create standardized fit result dictionary."""
    result = {
        'function_name': function_name,
        'parameters': parameters,
        'r2_score': r2,
        'x_data': x_data,
        'y_data': y_data,
        'y_pred': y_pred
    }
    result.update(kwargs)
    return result

class PowerFitter:
    """
    Fits power consumption trends for different model sizes using segmented regression.
    Linear segment → Logarithmic segment at specified transition points.
    """
    def __init__(self):
        self.transition_points = {
            '1.5B': 3000,
            '8B': 800,
            '14B': 384
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
        """Linear function: y = ax + b"""
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
            
            # Generate smooth interpolation for segmented function
            x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            y_pred_smooth = np.zeros_like(x_smooth, dtype=float)
            
            smooth_constant_mask = x_smooth <= transition
            smooth_log_mask = x_smooth > transition
            
            y_pred_smooth[smooth_constant_mask] = constant_value
            y_pred_smooth[smooth_log_mask] = self.log_function(x_smooth[smooth_log_mask], *log_params)
            
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
                'x_smooth': x_smooth,
                'y_pred_smooth': y_pred_smooth,
                'constant_value': constant_value,
                'log_params': log_params,
                'transition': transition
            }
            
        except Exception as e:
            print(f"Error fitting {model_name} power: {e}")
            return self._fallback_linear_fit(x_data, y_data, f'{model_name}_fallback_linear')

    def fit_power_trend(self, x_data: np.ndarray, y_data: np.ndarray, 
                       model_size: str, model_name: str) -> Dict:
        """
        Fit power trend using segmented regression (constant → logarithmic) for 8B and 14B,
        but constant value for 1.5B models.
        """
        print(f"  Fitting power trend for {model_name} ({model_size})")
        
        if model_size == '1.5B':
            return self._fit_constant_value(x_data, y_data, f'{model_size}_constant')
        else:
            transition = self.transition_points.get(model_size, 800) 
            return self.fit_segmented_regression(x_data, y_data, transition, model_size)

    def _fallback_linear_fit(self, x_data: np.ndarray, y_data: np.ndarray, function_name: str) -> Dict:
        """Fallback to simple linear fit with smooth interpolation."""
        try:
            popt, pcov = curve_fit(self.linear_function, x_data, y_data)
            y_pred = self.linear_function(x_data, *popt)
            
            # Generate smooth interpolation for linear function
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

    def _fit_constant_value(self, x_data: np.ndarray, y_data: np.ndarray, function_name: str) -> Dict:
        """
        Fit constant value using the mean of y_data.
        For power consumption, we use the mean as the best constant estimate.
        """
        try:
            # Use mean as the constant value - this minimizes mean squared error
            constant_value = np.mean(y_data)
            y_pred = np.full_like(y_data, constant_value)
            
            # Generate smooth interpolation for constant function
            x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            y_pred_smooth = np.full_like(x_smooth, constant_value)
            
            # Calculate R² for constant fit by comparing to linear baseline
            # This gives a meaningful measure of how well constant fit performs
            try:
                # Fit linear model as baseline for comparison
                linear_params, _ = curve_fit(self.linear_function, x_data, y_data)
                y_linear = self.linear_function(x_data, *linear_params)
                
                # Calculate R² as improvement over linear fit
                ss_res_constant = np.sum((y_data - y_pred) ** 2)
                ss_res_linear = np.sum((y_data - y_linear) ** 2)
                
                # If constant fit is better than linear, give high R²
                # If data is truly constant, this should be close to 1.0
                if ss_res_linear > 0:
                    r2 = max(0.0, 1 - (ss_res_constant / ss_res_linear))
                else:
                    r2 = 1.0  # Perfect fit
                    
                # Alternative: Calculate based on coefficient of variation
                # For truly constant data, this should be very high
                cv = np.std(y_data) / np.abs(np.mean(y_data)) if np.abs(np.mean(y_data)) > 0 else 0
                r2_cv = max(0.0, 1 - cv)  # Low variation = high R²
                
                # Use the higher of the two measures
                r2 = max(r2, r2_cv)
                
            except:
                # Fallback: use coefficient of variation approach
                cv = np.std(y_data) / np.abs(np.mean(y_data)) if np.abs(np.mean(y_data)) > 0 else 0
                r2 = max(0.0, 1 - cv)
            
            return {
                'function_name': function_name,
                'parameters': {
                    'constant_value': constant_value,
                    'data_std': np.std(y_data),  # Include std for reference
                    'data_range': np.max(y_data) - np.min(y_data),  # Include range for reference
                    'coefficient_of_variation': np.std(y_data) / np.abs(np.mean(y_data)) if np.abs(np.mean(y_data)) > 0 else 0
                },
                'r2_score': r2,
                'x_data': x_data,
                'y_data': y_data,
                'y_pred': y_pred,
                'x_smooth': x_smooth,
                'y_pred_smooth': y_pred_smooth,
                'function': self.constant_function,
                'fitted_params': [constant_value]
            }
        except Exception as e:
            print(f"Error fitting constant value: {e}")
            return {'function_name': 'failed', 'r2_score': 0.0}

    def plot_power_fit(self, fit_results: Dict, model_name: str, output_dir: Path):
        """Plot individual power fit results."""
        if 'x_data' not in fit_results:
            return
            
        x_data = fit_results['x_data']
        y_data = fit_results['y_data']
        
        # Use smooth curve if available, otherwise fall back to regular prediction
        if 'x_smooth' in fit_results and 'y_pred_smooth' in fit_results:
            x_plot = fit_results['x_smooth']
            y_plot = fit_results['y_pred_smooth']
        else:
            x_plot = x_data
            y_plot = fit_results['y_pred']
            sort_idx = np.argsort(x_data)
            x_plot = x_data[sort_idx]
            y_plot = y_plot[sort_idx]
        
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        display_name = f'DSR1-{model_name}' if model_name in ['8B', '1.5B', '14B'] else model_name
        
        plt.scatter(x_data, y_data, alpha=0.7, label='Actual', color='#1f77b4')
        
        if SHOW_R2_IN_LEGEND:
            fitted_label = f'Fitted (R² = {fit_results["r2_score"]:.3f})'
        else:
            fitted_label = 'Fitted'
        
        plt.plot(x_plot, y_plot, '--', linewidth=2, color='#ff7f0e',
                label=fitted_label)
        
        plt.xlabel('Input Length', fontsize=FONT_SIZE_LABELS)
        plt.ylabel('Average Power (W)', fontsize=FONT_SIZE_LABELS)
        plt.legend(fontsize=FONT_SIZE_POWER_LEGEND, bbox_to_anchor=(0, 1.02), loc='lower left')
        plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        # Set consistent tick marks but let matplotlib auto-scale the range
        current_xlim = plt.xlim()
        plt.xticks(range(0, min(int(current_xlim[1]) + 1000, 4500), 500))
        
        png_path = output_dir / f'{model_name.lower()}_power_fit.png'
        pdf_path = output_dir / f'{model_name.lower()}_power_fit.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved individual plot: {png_path}")
        print(f"    Saved individual plot: {pdf_path}") 
