"""
Scaling Results Parser - Modular analysis of test-time scaling results

A well-organized parser for processing scaling test results with separate modules for:
- Data processing
- Export functionality  
- Analysis and trends
- Utilities and data structures
"""

from .processors.core_parser import ScalingResultsProcessor
from .exporters.excel_exporter import ExcelExporter
from .utils.data_structures import ScalingRunMetadata, ScalingMetrics, ParsedResult

__version__ = "1.0.0"
__all__ = [
    "ScalingResultsProcessor",
    "ExcelExporter",
    "ScalingRunMetadata",
    "ScalingMetrics", 
    "ParsedResult"
] 