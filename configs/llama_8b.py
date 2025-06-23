"""
Configuration for LLaMA-8B model experiments.
"""

from pathlib import Path
import sys
sys.path.append('.')

from src.core.config import (
    ModelConfig, HardwareConfig, RegressionConfig, DataConfig, ExperimentConfig,
    ModelSize, HardwareType, RegressionType
)

# Model configuration
MODEL_CONFIG = ModelConfig(
    name="DeepSeek-R1-Distill-Llama-8B",
    size=ModelSize.MEDIUM,
    parameter_count="8B",
    expected_token_range={
        "min_input_tokens": 10,
        "max_input_tokens": 4000,
        "min_output_tokens": 1,
        "max_output_tokens": 2000
    }
)

# Hardware configurations
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
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },
    cross_validation_folds=5,
    test_size=0.2,
    random_state=42
)

# Data configuration
DATA_CONFIG = DataConfig(
    server_data_path=Path("datasets/server/full_mmlu_by_model.xlsx"),
    jetson_data_path=Path("datasets/tegra"),
    output_path=Path("outputs"),
    supported_models=["DeepSeek-R1-Distill-Llama-8B", "llama-8b", "llama_8b", "Llama-8B"]
)

# Complete experiment configurations
LLAMA_8B_LINEAR_EXPERIMENT = ExperimentConfig(
    model_config=MODEL_CONFIG,
    hardware_config=JETSON_HARDWARE,
    regression_config=LINEAR_REGRESSION_CONFIG,
    data_config=DATA_CONFIG,
    experiment_name="llama_8b_linear"
)

LLAMA_8B_POLYNOMIAL_EXPERIMENT = ExperimentConfig(
    model_config=MODEL_CONFIG,
    hardware_config=JETSON_HARDWARE,
    regression_config=POLYNOMIAL_REGRESSION_CONFIG,
    data_config=DATA_CONFIG,
    experiment_name="llama_8b_polynomial"
)

LLAMA_8B_RF_EXPERIMENT = ExperimentConfig(
    model_config=MODEL_CONFIG,
    hardware_config=JETSON_HARDWARE,
    regression_config=RANDOM_FOREST_CONFIG,
    data_config=DATA_CONFIG,
    experiment_name="llama_8b_rf"
)
