"""
Test configuration and core functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append('.')

from src.core.config import (
    ModelConfig, HardwareConfig, RegressionConfig, 
    ModelSize, HardwareType, RegressionType
)


class TestModelConfig:
    """Test cases for ModelConfig."""
    
    def test_valid_config(self):
        """Test creating a valid model configuration."""
        config = ModelConfig(
            name="TestModel",
            size=ModelSize.SMALL,
            parameter_count="1.5B",
            expected_token_range={
                "min_input_tokens": 1,
                "max_input_tokens": 1000,
                "min_output_tokens": 1,
                "max_output_tokens": 500
            }
        )
        assert config.name == "TestModel"
        assert config.size == ModelSize.SMALL
    
    def test_invalid_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ModelConfig(
                name="",
                size=ModelSize.SMALL,
                parameter_count="1.5B",
                expected_token_range={
                    "min_input_tokens": 1,
                    "max_input_tokens": 1000,
                    "min_output_tokens": 1,
                    "max_output_tokens": 500
                }
            )
    
    def test_invalid_token_range(self):
        """Test that invalid token range raises ValueError."""
        with pytest.raises(ValueError, match="Token range must contain keys"):
            ModelConfig(
                name="TestModel",
                size=ModelSize.SMALL,
                parameter_count="1.5B",
                expected_token_range={"invalid_key": 1}
            )


class TestHardwareConfig:
    """Test cases for HardwareConfig."""
    
    def test_valid_config(self):
        """Test creating a valid hardware configuration."""
        config = HardwareConfig(
            type=HardwareType.JETSON,
            name="Jetson-Orin",
            memory_gb=8,
            compute_capability="8.7"
        )
        assert config.type == HardwareType.JETSON
        assert config.memory_gb == 8
    
    def test_invalid_memory(self):
        """Test that negative memory raises ValueError."""
        with pytest.raises(ValueError, match="Memory must be positive"):
            HardwareConfig(
                type=HardwareType.JETSON,
                name="TestDevice",
                memory_gb=-1
            )


class TestRegressionConfig:
    """Test cases for RegressionConfig."""
    
    def test_valid_config(self):
        """Test creating a valid regression configuration."""
        config = RegressionConfig(
            type=RegressionType.LINEAR,
            hyperparameters={"fit_intercept": True},
            test_size=0.2,
            cross_validation_folds=5
        )
        assert config.type == RegressionType.LINEAR
        assert config.test_size == 0.2
    
    def test_invalid_test_size(self):
        """Test that invalid test size raises ValueError."""
        with pytest.raises(ValueError, match="Test size must be between 0 and 1"):
            RegressionConfig(
                type=RegressionType.LINEAR,
                hyperparameters={},
                test_size=1.5
            )
    
    def test_invalid_cv_folds(self):
        """Test that invalid CV folds raises ValueError."""
        with pytest.raises(ValueError, match="Cross validation folds must be at least 2"):
            RegressionConfig(
                type=RegressionType.LINEAR,
                hyperparameters={},
                cross_validation_folds=1
            )
