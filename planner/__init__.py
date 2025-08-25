"""
Planner Analysis Module.

Provides utilities for analyzing planner performance, comparing different
planning strategies, and generating performance plots and reports.
"""

from . import config
from . import plots
from . import processing

try:
    from . import analysis
    from . import compare
    from . import main
except ImportError:
    pass

__all__ = [
    "config",
    "plots", 
    "processing",
    "analysis",
    "compare",
    "main",
]
