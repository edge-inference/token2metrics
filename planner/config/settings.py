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
