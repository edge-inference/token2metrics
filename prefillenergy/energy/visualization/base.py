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
    
    def save_plot(self, filename: str, **kwargs) -> Dict[str, str]:
        pdf_path = PathManager.get_chart_path(f'{filename}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', **kwargs)
        
        plt.close()
        
        return {
            'pdf': pdf_path
        }
    
    def create_shared_legend(self, ax, fig, **kwargs):
        """Create a shared legend for multiple subplots."""
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title='Model', loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=len(labels), 
                  fontsize=10, title_fontsize=12, **kwargs)
        return handles, labels 