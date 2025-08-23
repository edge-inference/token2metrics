"""
Data structures for scaling results parsing.

Contains the core dataclasses used throughout the scaling parser.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class ScalingRunMetadata:
    """Metadata for a single scaling run."""
    run_dir: str
    model_name: str
    num_samples: int
    token_budget: int
    seed: int
    timestamp: str
    question_id: Optional[int] = None
    subject: Optional[str] = None
    input_tokens: Optional[int] = None


@dataclass
class ScalingMetrics:
    """Core scaling metrics for a run."""
    accuracy: float
    total_questions: int
    correct_answers: int
    avg_time_per_question: float
    avg_tokens_per_second: float
    total_samples_generated: int
    avg_voting_confidence: float
    scaling_efficiency: Optional[float] = None
    
    # Performance metrics
    avg_ttft: Optional[float] = None
    avg_decode_time: Optional[float] = None
    avg_input_tokens: Optional[float] = None
    avg_output_tokens: Optional[float] = None
    
    # Energy metrics (if available)
    avg_power_consumption: Optional[float] = None
    total_energy_consumed: Optional[float] = None


@dataclass
class ParsedResult:
    """Complete parsed result for a scaling run."""
    metadata: ScalingRunMetadata
    metrics: ScalingMetrics
    question_details: List[Dict[str, Any]] = field(default_factory=list)
    performance_data: Optional[pd.DataFrame] = None
    energy_data: Optional[pd.DataFrame] = None
    system_stats: Optional[Dict[str, Any]] = None 