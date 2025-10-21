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