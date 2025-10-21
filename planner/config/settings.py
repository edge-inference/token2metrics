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
Configuration settings for Natural-Plan planner.

Centralizes all hardcoded values scattered throughout the codebase.
"""

from pathlib import Path

# Task and model definitions
TASKS = ["meeting", "calendar", "trip"]
MODELS = ["14b", "8b", "1.5b"]

# Directory paths
PATHS = {
    # Input data directories
    "results_dir": "data/planner/server/",
    "reference_data": "benchmarks/agentic_planner/eval/data",
    
    # Evaluation type directories
    "baseline_results_dir": "data/planner/server/base",
    "budget_results_dir": "data/planner/server/budget", 
    "direct_results_dir": "data/planner/server/direct",
    "scaling_results_dir": "data/planner/server/scaling",
    
    # Output directories  
    "output_dir": "outputs/planner/",
    "plots_dir": "outputs/planner/plots/",
    "merged_results_dir": "data/planner/merged_results",
    "scored_results_dir": "outputs/planner/scored_results",
    
    # Benchmark paths
    "benchmark_root": "benchmarks/agentic_planner",
}

# Evaluation dataset mappings
EVAL_DATA_MAP = {
    "meeting": "benchmarks/agentic_planner/eval/data/meeting_planning.json",
    "calendar": "benchmarks/agentic_planner/eval/data/calendar_scheduling.json", 
    "trip": "benchmarks/agentic_planner/eval/data/trip_planning.json",
}

# Evaluation script mappings
EVAL_SCRIPTS = {
    "meeting": "benchmarks/agentic_planner/eval/evaluate_meeting_planning.py",
    "calendar": "benchmarks/agentic_planner/eval/evaluate_calendar_scheduling.py",
    "trip": "benchmarks/agentic_planner/eval/evaluate_trip_planning.py"
}

# Default CLI arguments
DEFAULT_ARGS = {
    "results_dir": PATHS["results_dir"],
    "output_dir": PATHS["output_dir"],
    "reference_data": PATHS["reference_data"],
    "models": ",".join(MODELS),
    "tasks": TASKS,
}

# Token budget limits
TOKEN_BUDGETS = [128, 256, 512, 1024]

# Plot styling
PLOT_CONFIG = {
    "figsize": (12, 8),
    "dpi": 300,
    "style": "default",
    "palette": "husl",
    "markers": ['o', 's', '^', 'D', 'v', '<', '>', 'p'],
}

# File patterns
FILE_PATTERNS = {
    "results_json": "results_*.json",
    "detailed_csv": "detailed_results_*.csv", 
    "log_files": "*.log",
    "task_dir_pattern": r"([^_]+)_(\d{8}_\d{6})",  # task_YYYYMMDD_HHMMSS
}
