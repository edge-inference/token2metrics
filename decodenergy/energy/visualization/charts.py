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
Main chart components for energy analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any
from .base import BasePlotter


class PowerScalingCharts(BasePlotter):
    """Creates power scaling analysis charts."""
    
    def create(self, analysis_df: pd.DataFrame) -> str:
        """Create power scaling visualization with 3 subplots."""
        if analysis_df.empty:
            return ""
        
        self.setup_plot_style()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 4))
        
        for model in analysis_df['model_name'].unique():
            model_data = analysis_df[analysis_df['model_name'] == model].sort_values('target_token_range')
            x = model_data['target_token_range']
            y = model_data['avg_power_w_mean']
            
            ax1.scatter(x, y, marker='o', s=60, label=model, alpha=0.8)
            
            if len(x) >= 3:
                import numpy as np
                z = np.polyfit(x, y, min(2, len(x)-1))
                p = np.poly1d(z)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax1.plot(x_smooth, p(x_smooth), '--', alpha=0.6)
        
        ax1.set_xlabel('Output Length', fontsize=8)
        ax1.set_ylabel('Average Power (W)', fontsize=8)
        ax1.legend().set_visible(False)
        ax1.tick_params(axis='both', which='major', labelsize=6)
        
        for model in analysis_df['model_name'].unique():
            model_data = analysis_df[analysis_df['model_name'] == model].sort_values('target_token_range')
            x = model_data['target_token_range']
            y = model_data['energy_per_token_mean']
            
            ax2.scatter(x, y, marker='o', s=60, label=model, alpha=0.8)
            
            if len(x) >= 3:
                import numpy as np
                z = np.polyfit(x, y, min(2, len(x)-1))
                p = np.poly1d(z)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax2.plot(x_smooth, p(x_smooth), '--', alpha=0.6)
        
        ax2.set_xlabel('Output Length', fontsize=8)
        ax2.set_ylabel('Energy (J/token)', fontsize=8)
        ax2.legend().set_visible(False)
        ax2.tick_params(axis='both', which='major', labelsize=6)
        
        for model in analysis_df['model_name'].unique():
            model_data = analysis_df[analysis_df['model_name'] == model].sort_values('target_token_range')
            x = model_data['target_token_range']
            y = model_data['tokens_per_second_mean']
            
            ax3.scatter(x, y, marker='o', s=60, label=model, alpha=0.8)
            
            if len(x) >= 3:
                import numpy as np
                z = np.polyfit(x, y, min(2, len(x)-1))
                p = np.poly1d(z)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax3.plot(x_smooth, p(x_smooth), '--', alpha=0.6)
        
        ax3.set_xlabel('Output Length', fontsize=8)
        ax3.set_ylabel('Output Tokens per Second', fontsize=8)
        ax3.legend().set_visible(False)
        ax3.tick_params(axis='both', which='major', labelsize=6)
        
        self.create_shared_legend(ax1, fig)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        paths = self.save_plot('power_scaling_analysis')
        print(f"Power scaling chart (PDF): {paths['pdf']}")
        return paths['pdf']


class EfficiencyHeatmap(BasePlotter):
    """Creates efficiency heatmap visualizations."""
    
    def create(self, filtered_df: pd.DataFrame) -> str:
        """Create efficiency heatmap visualization."""
        if filtered_df.empty:
            return ""
        
        self.setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))
        
        # Heatmap 1: Average Power
        power_pivot = filtered_df.pivot_table(
            index='model_name', 
            columns='target_token_range', 
            values='avg_power_w', 
            aggfunc='mean'
        )
        
        sns.heatmap(power_pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': 'Average Power (W)'})
        ax1.set_xlabel('Decode Token Range', fontsize=12)
        ax1.set_ylabel('Model', fontsize=12)
        
        # Heatmap 2: Energy per Token
        energy_pivot = filtered_df.pivot_table(
            index='model_name', 
            columns='target_token_range', 
            values='energy_per_token', 
            aggfunc='mean'
        )
        
        sns.heatmap(energy_pivot, annot=True, fmt='.3f', cmap='viridis', 
                   ax=ax2, cbar_kws={'label': 'Energy (J/token)'})
        ax2.set_xlabel('Decode Token Range', fontsize=12)
        ax2.set_ylabel('Model', fontsize=12)
        
        plt.tight_layout()
        
        # Save and return
        paths = self.save_plot('efficiency_heatmap')
        print(f"Efficiency heatmap (PDF): {paths['pdf']}")
        return paths['pdf'] 