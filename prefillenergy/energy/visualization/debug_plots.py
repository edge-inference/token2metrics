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
Debug plotting components for detailed trend analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base import BasePlotter
from ..utils import sort_models_by_size


class DebugPlotter(BasePlotter):
    """Creates debug plots with more granular data points."""
    
    def __init__(self):
        super().__init__()
        self.debug_token_ranges = [3, 24, 50, 75, 100, 128, 150, 175, 200, 225, 256, 
                                  300, 350, 384, 450, 512, 575, 640, 700, 1013]
        self.tolerance = 15
    
    def create(self, model_data: Dict[str, pd.DataFrame]) -> str:
        """Create debug plots with extended data points."""
        if not model_data:
            return ""
        
        # Get debug data
        debug_data = self._get_debug_data(model_data)
        if debug_data.empty:
            print("No debug data found for extended token ranges")
            return ""
        
        # Create debug visualization
        return self._create_debug_visualization(debug_data)
    
    def _get_debug_data(self, model_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Filter and aggregate data for debug token ranges."""
        debug_results = []
        
        for model_name, df in model_data.items():
            for target_range in self.debug_token_ranges:
                # Filter questions within tolerance
                filtered = df[
                    (df['input_tokens'] >= target_range - self.tolerance) & 
                    (df['input_tokens'] <= target_range + self.tolerance)
                ]
                
                if not filtered.empty:
                    # Calculate aggregated metrics
                    debug_results.append({
                        'model_name': model_name,
                        'target_token_range': target_range,
                        'avg_power_w': filtered['avg_power_w'].mean(),
                        'energy_per_token': filtered['energy_per_token'].mean(),
                        'tokens_per_second': (filtered['output_tokens'] / 
                                            (filtered['total_time_ms'] / 1000)).mean(),
                        'question_count': len(filtered)
                    })
        
        return pd.DataFrame(debug_results)
    
    def _create_debug_visualization(self, debug_data: pd.DataFrame) -> str:
        """Create debug visualization with subplots."""
        self.setup_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Average Power with more data points
        self._plot_debug_metric(debug_data, 'avg_power_w', 'Average Power (W)', 
                               axes[0, 0], 'o-')
        
        # Subplot 2: Energy Efficiency with more data points
        self._plot_debug_metric(debug_data, 'energy_per_token', 'Energy (J/token)', 
                               axes[0, 1], '^-')
        
        # Subplot 3: Processing Speed with more data points
        self._plot_debug_metric(debug_data, 'tokens_per_second', 'Output Tokens per Second', 
                               axes[1, 0], 'd-')
        
        # Subplot 4: Question Count Distribution
        self._plot_question_distribution(debug_data, axes[1, 1])
        
        # Shared legend
        self.create_shared_legend(axes[0, 0], fig)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # Save and return
        paths = self.save_plot('debug_analysis')
        print(f"Debug chart (PDF): {paths['pdf']}")
        return paths['pdf']
    
    def _plot_debug_metric(self, debug_data: pd.DataFrame, metric: str, 
                          ylabel: str, ax, marker_style: str):
        """Plot a single metric with debug data points."""
        for model in sort_models_by_size(debug_data['model_name'].unique()):
            model_data = debug_data[debug_data['model_name'] == model]
            ax.plot(model_data['target_token_range'], model_data[metric], 
                   marker_style, label=model, linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_xlabel('Prefill Tokens (Debug Ranges)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend().set_visible(False)
        
        # Set x-axis to show all debug ranges
        ax.set_xticks(self.debug_token_ranges[::2])  # Show every other tick
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_question_distribution(self, debug_data: pd.DataFrame, ax):
        """Plot question count distribution across token ranges."""
        question_counts = debug_data.groupby('target_token_range')['question_count'].sum()
        
        ax.bar(question_counts.index, question_counts.values, 
               alpha=0.7, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Prefill Token Range', fontsize=12)
        ax.set_ylabel('Total Questions', fontsize=12)
        
        # Rotate x-axis labels
        ax.set_xticks(self.debug_token_ranges[::2])
        ax.tick_params(axis='x', rotation=45) 