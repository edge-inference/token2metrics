"""
Token2Metrics - Translating tokens to latency/energy metrics for edge inference.

This package provides utilities for:
- Prefill energy/latency modeling
- Decode energy/latency modeling  
- Scaling analysis
- Token-based performance predictions
"""

__version__ = "0.1.0"
__author__ = "Benjamin Kubwimana, Qijing Huang"

# Import main modules
from . import config
from . import decodenergy
from . import prefillenergy
from . import scalingenergy
from . import planner

__all__ = [
    "config",
    "decodenergy", 
    "prefillenergy",
    "scalingenergy",
    "planner",
]
