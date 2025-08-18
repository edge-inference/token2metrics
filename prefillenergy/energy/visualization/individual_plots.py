"""
Individual plot components for single metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from .base import BasePlotter
from ..utils import sort_models_by_size


class IndividualPlotter(BasePlotter):
    """Creates individual plots for each metric without titles."""
    
    def create_all(self, analysis_df: pd.DataFrame) -> Dict[str, str]:
        """Create individual plots for all metrics."""
        if analysis_df.empty:
            return {}
        
        plot_paths = {}
        
        plot_paths['avg_power'] = self._create_avg_power_plot(analysis_df)
        plot_paths['energy_efficiency'] = self._create_energy_efficiency_plot(analysis_df)
        plot_paths['tokens_per_second'] = self._create_tokens_per_second_plot(analysis_df)
        
        self._print_summary(plot_paths)
        return plot_paths
    
    def _create_avg_power_plot(self, analysis_df: pd.DataFrame) -> str:
        """Create average power plot with polynomial trends."""
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for model in sort_models_by_size(analysis_df['model_name'].unique()):
            model_data = analysis_df[analysis_df['model_name'] == model].sort_values('target_token_range')
            x = model_data['target_token_range']
            y = model_data['avg_power_w_mean']
            
            ax.scatter(x, y, marker='o', s=80, label=model, alpha=0.8)
            
            if len(x) >= 3:
                import numpy as np
                z = np.polyfit(x, y, min(2, len(x)-1))
                p = np.poly1d(z)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_smooth, p(x_smooth), '--', alpha=0.6)
        
        ax.set_xlabel('Prefill Tokens', fontsize=12)
        ax.set_ylabel('Average Power (W)', fontsize=12)
        ax.legend(title='Model', fontsize=12, title_fontsize=12)
        plt.tight_layout()
        
        paths = self.save_plot('avg_power_per_token')
        return paths['pdf']
    
    def _create_energy_efficiency_plot(self, analysis_df: pd.DataFrame) -> str:
        """Create energy efficiency plot with polynomial trends."""
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for model in sort_models_by_size(analysis_df['model_name'].unique()):
            model_data = analysis_df[analysis_df['model_name'] == model].sort_values('target_token_range')
            x = model_data['target_token_range']
            y = model_data['energy_per_token_mean']
            
            ax.scatter(x, y, marker='o', s=80, label=model, alpha=0.8)
            
            if len(x) >= 3:
                import numpy as np
                z = np.polyfit(x, y, min(2, len(x)-1))
                p = np.poly1d(z)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_smooth, p(x_smooth), '--', alpha=0.6)
        
        ax.set_xlabel('Prefill Tokens', fontsize=12)
        ax.set_ylabel('Energy (J/token)', fontsize=12)
        ax.legend(title='Model', fontsize=12, title_fontsize=12)
        plt.tight_layout()
        
        paths = self.save_plot('energy_efficiency_per_token')
        return paths['pdf']
    
    def _create_tokens_per_second_plot(self, analysis_df: pd.DataFrame) -> str:
        """Create output tokens per second plot with polynomial trends."""
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for model in sort_models_by_size(analysis_df['model_name'].unique()):
            model_data = analysis_df[analysis_df['model_name'] == model].sort_values('target_token_range')
            x = model_data['target_token_range']
            y = model_data['tokens_per_second_mean']
            
            ax.scatter(x, y, marker='o', s=80, label=model, alpha=0.8)
            
            if len(x) >= 3:
                import numpy as np
                z = np.polyfit(x, y, min(2, len(x)-1))
                p = np.poly1d(z)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_smooth, p(x_smooth), '--', alpha=0.6)
        
        ax.set_xlabel('Prefill Tokens', fontsize=12)
        ax.set_ylabel('Output Tokens per Second', fontsize=12)
        ax.legend(title='Model', fontsize=12, title_fontsize=12)
        plt.tight_layout()
        
        paths = self.save_plot('tokens_per_second')
        return paths['pdf']
    
    def _print_summary(self, plot_paths: Dict[str, str]):
        """Print summary of created plots."""
        print(f"Individual plots saved:")
        for plot_type, path in plot_paths.items():
            print(f"  {plot_type.replace('_', ' ').title()}: {path}") 