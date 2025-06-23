"""
Persistence utilities for saving and loading trained predictors.
"""
import joblib
from pathlib import Path
from typing import Any

def save_predictor(predictor: Any, model_size: str, phase: str, output_dir: Path) -> Path:
    """
    Save a trained predictor to disk using joblib.
    Args:
        predictor: Trained PhaseSpecificLatencyPredictor
        model_size: Model size string (e.g., '14B')
        phase: 'decode'
        output_dir: Directory to save model
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"predictor_{model_size}_{phase}.joblib"
    joblib.dump(predictor, file_path)
    return file_path

def load_predictor(model_size: str, phase: str, model_dir: Path) -> Any:
    """
    Load a trained predictor from disk.
    Args:
        model_size: Model size string (e.g., '14B')
        phase: 'decode'
        model_dir: Directory containing saved model
    Returns:
        Loaded PhaseSpecificLatencyPredictor
    """
    file_path = Path(model_dir) / f"predictor_{model_size}_{phase}.joblib"
    if not file_path.exists():
        raise FileNotFoundError(f"Predictor file not found: {file_path}")
    return joblib.load(file_path)
