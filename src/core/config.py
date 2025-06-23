"""
Core configuration data classes for token-based inference metrics modeling.

This module defines the configuration objects for different model sizes,
hardware specifications, and modeling parameters following the dataclass
pattern for immutable configuration objects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from pathlib import Path


class ModelSize(Enum):
    """Supported LLM model sizes."""
    SMALL = "1.5B"
    MEDIUM = "8B" 
    LARGE = "14B"


class HardwareType(Enum):
    """Supported hardware types for inference."""
    SERVER = "server"
    JETSON = "jetson"


class RegressionType(Enum):
    """Supported regression model types."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class CalibrationMethod(Enum):
    """Supported calibration methods."""
    SIMPLE_SCALING = "simple_scaling"
    LINEAR_FIT = "linear_fit"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific LLM model."""
    
    name: str
    size: ModelSize
    parameter_count: str
    expected_token_range: Dict[str, int]
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Model name cannot be empty")
        
        required_keys = {"min_input_tokens", "max_input_tokens", "min_output_tokens", "max_output_tokens"}
        if not required_keys.issubset(self.expected_token_range.keys()):
            raise ValueError(f"Token range must contain keys: {required_keys}")


@dataclass(frozen=True)
class HardwareConfig:
    """Configuration for hardware specifications."""
    
    type: HardwareType
    name: str
    memory_gb: int
    compute_capability: Optional[str] = None
    power_limit_w: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate hardware configuration."""
        if self.memory_gb <= 0:
            raise ValueError("Memory must be positive")


@dataclass(frozen=True)
class RegressionConfig:
    """Configuration for regression modeling."""
    
    type: RegressionType
    hyperparameters: Dict[str, Union[int, float, str]]
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    def __post_init__(self) -> None:
        """Validate regression configuration."""
        if not 0 < self.test_size < 1:
            raise ValueError("Test size must be between 0 and 1")
        if self.cross_validation_folds < 2:
            raise ValueError("Cross validation folds must be at least 2")


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and processing."""
    
    server_data_path: Path
    jetson_data_path: Path
    output_path: Path
    supported_models: List[str]
    
    def __post_init__(self) -> None:
        """Validate data paths."""
        if not self.server_data_path.exists():
            raise FileNotFoundError(f"Server data path not found: {self.server_data_path}")
        if not self.jetson_data_path.exists():
            raise FileNotFoundError(f"Jetson data path not found: {self.jetson_data_path}")


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for model calibration."""
    
    method: CalibrationMethod
    manual_scaling_factor: Optional[float] = None
    # Add other calibration parameters here if needed in the future


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration combining all components."""
    
    model_config: ModelConfig
    hardware_config: HardwareConfig
    regression_config: RegressionConfig
    data_config: DataConfig
    calibration_config: CalibrationConfig
    experiment_name: str
    
    def get_output_prefix(self) -> str:
        """Generate output file prefix based on configuration."""
        return f"{self.experiment_name}_{self.model_config.size.value}_{self.hardware_config.type.value}"
