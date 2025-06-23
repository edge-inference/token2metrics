"""
Utility functions and logging configuration for the modeling framework.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


def setup_logging(log_level: str = "INFO", 
                 log_file: str = None,
                 log_format: str = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger("token2metrics")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_experiment_results(results: Dict[str, Any], 
                           output_path: Path,
                           experiment_name: str) -> Path:
    """
    Save experiment results to JSON file with timestamp.
    
    Args:
        results: Dictionary containing experiment results
        output_path: Directory to save results
        experiment_name: Name of the experiment
        
    Returns:
        Path to saved results file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_results_{timestamp}.json"
    filepath = output_path / filename
    
    # Add metadata
    results_with_metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(results_with_metadata, f, indent=2, default=str)
    
    return filepath


def load_experiment_results(filepath: Path) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary containing experiment results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def validate_token_inputs(input_tokens: int, output_tokens: int) -> None:
    """
    Validate token input parameters.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Raises:
        ValueError: If token counts are invalid
    """
    if input_tokens <= 0:
        raise ValueError(f"Input tokens must be positive, got {input_tokens}")
    
    if output_tokens <= 0:
        raise ValueError(f"Output tokens must be positive, got {output_tokens}")
    
    if input_tokens > 10000:
        logging.warning(f"Input tokens ({input_tokens}) is very large, "
                       "prediction may be unreliable")
    
    if output_tokens > 5000:
        logging.warning(f"Output tokens ({output_tokens}) is very large, "
                       "prediction may be unreliable")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def create_output_directories(base_path: Path, experiment_name: str) -> Dict[str, Path]:
    """
    Create output directory structure for experiment.
    
    Args:
        base_path: Base output directory
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary mapping directory names to paths
    """
    experiment_dir = base_path / experiment_name
    
    directories = {
        "base": experiment_dir,
        "models": experiment_dir / "models",
        "results": experiment_dir / "results", 
        "plots": experiment_dir / "plots",
        "logs": experiment_dir / "logs"
    }
    
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return directories


class ExperimentTracker:
    """Simple experiment tracking utility."""
    
    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = None
        self.metrics = {}
        self.config = {}
    
    def start_experiment(self, config: Dict[str, Any]) -> None:
        """Start tracking an experiment."""
        self.start_time = datetime.now()
        self.config = config
        self.metrics = {}
        
        logging.info(f"Started experiment: {self.experiment_name}")
    
    def log_metric(self, name: str, value: float, step: int = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {"value": value, "timestamp": datetime.now().isoformat()}
        if step is not None:
            metric_entry["step"] = step
            
        self.metrics[name].append(metric_entry)
    
    def finish_experiment(self) -> Path:
        """Finish experiment and save results."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        results = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "config": self.config,
            "metrics": self.metrics
        }
        
        output_path = save_experiment_results(
            results, self.output_dir, self.experiment_name
        )
        
        logging.info(f"Finished experiment: {self.experiment_name} "
                    f"(Duration: {format_duration(duration)})")
        
        return output_path
