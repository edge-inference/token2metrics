"""
Token2Metrics: Analytical modeling framework for token-based inference metrics.

This package provides tools for predicting inference latency, energy, and power
consumption on edge devices (particularly Jetson) based on token counts from
server-side measurements.
"""

__version__ = "0.1.0"
__author__ = "Token2Metrics Team"

from .core.config import ModelSize, HardwareType, RegressionType
from .core.interfaces import ModelMetrics, PredictionResult

__all__ = [
    "ModelSize",
    "HardwareType", 
    "RegressionType",
    "ModelMetrics",
    "PredictionResult"
]
