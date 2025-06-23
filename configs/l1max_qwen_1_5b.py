"""
Configuration for L1Max (L1-Qwen-1.5B) model experiments.
"""

from pathlib import Path
import sys
sys.path.append('.')

from src.core.config import (
    ModelConfig, HardwareConfig, RegressionConfig, DataConfig, ExperimentConfig,
    ModelSize, HardwareType, RegressionType, CalibrationConfig, CalibrationMethod
)

# Model configuration
MODEL_CONFIG = ModelConfig(
    name="L1-Qwen-1_5B-Max",  
    size=ModelSize.SMALL,
    parameter_count="1.5B",
    expected_token_range={
        "min_input_tokens": 10,
        "max_input_tokens": 4096,
        "min_output_tokens": 1,
        "max_output_tokens": 32768
    }
)

# Hardware configurations
SERVER_HARDWARE = HardwareConfig(
    type=HardwareType.SERVER,
    name="Server-GPU",
    memory_gb=32,
    compute_capability="8.6"
)

JETSON_HARDWARE = HardwareConfig(
    type=HardwareType.JETSON,
    name="Jetson-Orin",
    memory_gb=8,
    compute_capability="8.7",
    power_limit_w=50
)

# Regression configurations
LINEAR_REGRESSION_CONFIG = RegressionConfig(
    type=RegressionType.LINEAR,
    hyperparameters={
        "fit_intercept": True,
        "normalize": False
    },
    cross_validation_folds=5,
    test_size=0.2,
    random_state=42
)

POLYNOMIAL_REGRESSION_CONFIG = RegressionConfig(
    type=RegressionType.POLYNOMIAL,
    hyperparameters={
        "degree": 2,
        "interaction_only": False,
        "include_bias": True
    },
    cross_validation_folds=5,
    test_size=0.2,
    random_state=42
)

RANDOM_FOREST_CONFIG = RegressionConfig(
    type=RegressionType.RANDOM_FOREST,
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },
    cross_validation_folds=5,
    test_size=0.2,
    random_state=42
)

# Calibration configuration
CALIBRATION_CONFIG = CalibrationConfig(
    method=CalibrationMethod.SIMPLE_SCALING
)

# Data configuration
DATA_CONFIG = DataConfig(
    server_data_path=Path("datasets/server/full_mmlu_by_model.xlsx"),
    jetson_data_path=Path("datasets/tegra"),
    output_path=Path("outputs"),
    supported_models=["L1-Qwen-1_5B-Max", "L1Max-Qwen-1_5B", "l1max-qwen-1.5b", "l1max_qwen_1_5b"]
)

# Complete experiment configurations
L1MAX_LINEAR_EXPERIMENT = ExperimentConfig(
    model_config=MODEL_CONFIG,
    hardware_config=JETSON_HARDWARE,
    regression_config=LINEAR_REGRESSION_CONFIG,
    data_config=DATA_CONFIG,
    calibration_config=CALIBRATION_CONFIG,
    experiment_name="l1max_qwen_1_5b_linear"
)

L1MAX_POLYNOMIAL_EXPERIMENT = ExperimentConfig(
    model_config=MODEL_CONFIG,
    hardware_config=JETSON_HARDWARE,
    regression_config=POLYNOMIAL_REGRESSION_CONFIG,
    data_config=DATA_CONFIG,
    calibration_config=CALIBRATION_CONFIG,
    experiment_name="l1max_qwen_1_5b_polynomial"
)

L1MAX_RF_EXPERIMENT = ExperimentConfig(
    model_config=MODEL_CONFIG,
    hardware_config=JETSON_HARDWARE,
    regression_config=RANDOM_FOREST_CONFIG,
    data_config=DATA_CONFIG,
    calibration_config=CALIBRATION_CONFIG,
    experiment_name="l1max_qwen_1_5b_rf"
)
