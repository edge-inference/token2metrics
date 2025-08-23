"""
Base visualization components for energy analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from ..utils import PathManager


class BasePlotter:
    """Base class for all energy visualization components."""
    
    def __init__(self):
        """Initialize base plotter with white background style."""
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Configure matplotlib for white background plots."""
        plt.style.use('default')
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
    
    def get_timestamp(self) -> str:
        """Get formatted timestamp for file naming."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_plot(self, filename: str, save_png: bool = False, **kwargs) -> Dict[str, str]:
        """Save plot in PDF format (and optionally PNG)."""
        result = {}
        
        pdf_path = PathManager.get_chart_path(f'{filename}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', **kwargs)
        result['pdf'] = pdf_path
        
        if save_png:
            png_path = PathManager.get_chart_path(f'{filename}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', **kwargs)
            result['png'] = png_path
        
        plt.close()
        return result
    
    def create_shared_legend(self, ax, fig, **kwargs):
        """Create a shared legend for multiple subplots."""
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title='Model', loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=len(labels), 
                  fontsize=10, title_fontsize=12, **kwargs)
        return handles, labels 