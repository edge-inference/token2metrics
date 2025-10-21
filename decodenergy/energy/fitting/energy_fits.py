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
Energy per decode token fitting functions for different model sizes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

FIGURE_WIDTH = 4
FIGURE_HEIGHT = 4
FONT_SIZE_LABELS = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_TICKS = 10
SHOW_R2_IN_LEGEND = False

class EnergyFitter:
    """Handles energy per decode token fitting for different model sizes."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def logarithmic_function(x, a, b):
        """Logarithmic function: y = a*ln(x) + b"""
        return a * np.log(x) + b
    

    
    def fit_14b_energy(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 14B model energy per decode token: logarithmic fit."""
        try:
            popt, pcov = curve_fit(self.logarithmic_function, x_data, y_data)
            y_pred = self.logarithmic_function(x_data, *popt)
            
            x_min = 0.1
            x_max = max(x_data.max(), 2048)
            x_smooth = np.linspace(x_min, x_max, 200)
            y_smooth = self.logarithmic_function(x_smooth, *popt)
            y_smooth = np.maximum(y_smooth, 0)
            
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
                'function': self.logarithmic_function,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 14B energy (logarithmic): {e}")
            raise RuntimeError(f"Failed to fit 14B energy with logarithmic function: {e}")
    
    def fit_8b_energy(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 8B model energy per decode token: logarithmic fit."""
        try:
            popt, pcov = curve_fit(self.logarithmic_function, x_data, y_data)
            y_pred = self.logarithmic_function(x_data, *popt)
            
            x_min = 0.1
            x_max = max(x_data.max(), 2048)
            x_smooth = np.linspace(x_min, x_max, 200)
            y_smooth = self.logarithmic_function(x_smooth, *popt)
            y_smooth = np.maximum(y_smooth, 0)
            
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
                'function': self.logarithmic_function,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 8B energy (logarithmic): {e}")
            raise RuntimeError(f"Failed to fit 8B energy with logarithmic function: {e}")
    
    def fit_1_5b_energy(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Fit 1.5B model energy per decode token: logarithmic fit."""
        try:
            popt, pcov = curve_fit(self.logarithmic_function, x_data, y_data)
            y_pred = self.logarithmic_function(x_data, *popt)
            
            x_min = 0.1
            x_max = max(x_data.max(), 2048)
            x_smooth = np.linspace(x_min, x_max, 200)
            y_smooth = self.logarithmic_function(x_smooth, *popt)
            y_smooth = np.maximum(y_smooth, 0)
            
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
                'function': self.logarithmic_function,
                'fitted_params': popt
            }
        except Exception as e:
            print(f"Error fitting 1.5B energy (logarithmic): {e}")
            raise RuntimeError(f"Failed to fit 1.5B energy with logarithmic function: {e}")
    

    
    def fit_energy_trend(self, x_data: np.ndarray, y_data: np.ndarray, 
                        model_size: str, model_name: str) -> Dict:
        """
        Fit energy per decode token trend based on model size.
        
        Args:
            x_data: Decode token counts
            y_data: Energy per token values
            model_size: Detected model size ('1.5B', '8B', '14B', etc.)
            model_name: Full model name for logging
            
        Returns:
            Dictionary with fitting results
        """
        print(f"  Fitting energy per decode token trend for {model_name} ({model_size})")
        if model_size == '14B':
            return self.fit_14b_energy(x_data, y_data)
        elif model_size == '8B':
            return self.fit_8b_energy(x_data, y_data)
        elif model_size == '1.5B':
            return self.fit_1_5b_energy(x_data, y_data)
        else:
            print(f"  Unknown model size {model_size}, no logarithmic fit available")
            raise RuntimeError(f"No logarithmic fitting function available for model size: {model_size}")
    
    def plot_energy_fit(self, fit_results: Dict, model_name: str, output_dir: Path):
        """Plot individual energy per decode token fit results with smooth logarithmic curves."""
        if 'x_data' not in fit_results:
            return
        
        x_data = fit_results['x_data']
        y_data = fit_results['y_data']
        x_smooth = fit_results.get('x_smooth', x_data)
        y_smooth = fit_results.get('y_smooth', fit_results.get('y_pred', y_data))
        
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        display_name = f'DSR1-{model_name}' if model_name in ['8B', '1.5B', '14B'] else model_name
        
        # Plot data points
        ax.scatter(x_data, y_data, alpha=0.7, color='#1f77b4', label='Actual')
        
        if SHOW_R2_IN_LEGEND:
            fitted_label = f'Fitted (R² = {fit_results["r2_score"]:.4f})'
        else:
            fitted_label = 'Fitted'
        
        ax.plot(x_smooth, y_smooth, '--', linewidth=2, color='#ff7f0e', label=fitted_label)
        
        ax.set_xlabel('Output Length', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy (J/token)', fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left', bbox_to_anchor=(0.02, 1.02))
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
        
        import numpy as np
        x_min, x_max = ax.get_xlim()
        tick_start = int(np.ceil(x_min / 128) * 128)
        tick_end = int(np.floor(x_max / 128) * 128) + 1
        ticks = np.arange(tick_start, tick_end, 128)
        ax.set_xticks(ticks)
        plt.tight_layout()
        safe_name = model_name.replace('/', '_').replace('\\', '_')
        pdf_path = output_dir / f'{safe_name}_energy_fit.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"    Saved energy fit plot to: {pdf_path}")
        plt.close()
