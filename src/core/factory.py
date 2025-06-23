"""
Factory classes for creating model instances following the Factory pattern.

This module provides factories for instantiating different types of regression
models, data loaders, and preprocessors based on configuration.
"""

from typing import Dict, Type
import logging

from ..core.interfaces import RegressionStrategy, DataLoader, DataPreprocessor, ModelCalibrator
from ..core.config import RegressionType, HardwareType, ModelSize

logger = logging.getLogger(__name__)


class RegressionFactory:
    """Factory for creating regression strategy instances."""
    
    _strategies: Dict[RegressionType, Type[RegressionStrategy]] = {}
    
    @classmethod
    def register_strategy(cls, 
                         regression_type: RegressionType, 
                         strategy_class: Type[RegressionStrategy]) -> None:
        """
        Register a new regression strategy.
        
        Args:
            regression_type: Type of regression to register
            strategy_class: Class implementing the strategy
        """
        cls._strategies[regression_type] = strategy_class
        logger.info(f"Registered regression strategy: {regression_type.value}")
    
    @classmethod
    def create_strategy(cls, 
                       regression_type: RegressionType,
                       **kwargs) -> RegressionStrategy:
        """
        Create a regression strategy instance.
        
        Args:
            regression_type: Type of regression to create
            **kwargs: Additional arguments for strategy construction
            
        Returns:
            Configured regression strategy instance
            
        Raises:
            ValueError: If regression type is not registered
        """
        if regression_type not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown regression type: {regression_type}. "
                           f"Available types: {available}")
        
        strategy_class = cls._strategies[regression_type]
        return strategy_class(**kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> list[RegressionType]:
        """Get list of available regression strategies."""
        return list(cls._strategies.keys())


class DataLoaderFactory:
    """Factory for creating data loader instances."""
    
    _loaders: Dict[str, Type[DataLoader]] = {}
    
    @classmethod
    def register_loader(cls, 
                       loader_name: str, 
                       loader_class: Type[DataLoader]) -> None:
        """
        Register a new data loader.
        
        Args:
            loader_name: Name identifier for the loader
            loader_class: Class implementing the data loader
        """
        cls._loaders[loader_name] = loader_class
        logger.info(f"Registered data loader: {loader_name}")
    
    @classmethod
    def create_loader(cls, 
                     loader_name: str,
                     **kwargs) -> DataLoader:
        """
        Create a data loader instance.
        
        Args:
            loader_name: Name of the loader to create
            **kwargs: Additional arguments for loader construction
            
        Returns:
            Configured data loader instance
            
        Raises:
            ValueError: If loader name is not registered
        """
        if loader_name not in cls._loaders:
            available = list(cls._loaders.keys())
            raise ValueError(f"Unknown loader: {loader_name}. "
                           f"Available loaders: {available}")
        
        loader_class = cls._loaders[loader_name]
        return loader_class(**kwargs)


class PreprocessorFactory:
    """Factory for creating data preprocessor instances."""
    
    _preprocessors: Dict[str, Type[DataPreprocessor]] = {}
    
    @classmethod
    def register_preprocessor(cls, 
                             preprocessor_name: str, 
                             preprocessor_class: Type[DataPreprocessor]) -> None:
        """
        Register a new data preprocessor.
        
        Args:
            preprocessor_name: Name identifier for the preprocessor
            preprocessor_class: Class implementing the preprocessor
        """
        cls._preprocessors[preprocessor_name] = preprocessor_class
        logger.info(f"Registered preprocessor: {preprocessor_name}")
    
    @classmethod
    def create_preprocessor(cls, 
                           preprocessor_name: str,
                           **kwargs) -> DataPreprocessor:
        """
        Create a preprocessor instance.
        
        Args:
            preprocessor_name: Name of the preprocessor to create
            **kwargs: Additional arguments for preprocessor construction
            
        Returns:
            Configured preprocessor instance
            
        Raises:
            ValueError: If preprocessor name is not registered
        """
        if preprocessor_name not in cls._preprocessors:
            available = list(cls._preprocessors.keys())
            raise ValueError(f"Unknown preprocessor: {preprocessor_name}. "
                           f"Available preprocessors: {available}")
        
        preprocessor_class = cls._preprocessors[preprocessor_name]
        return preprocessor_class(**kwargs)


class CalibratorFactory:
    """Factory for creating model calibrator instances."""
    
    _calibrators: Dict[str, Type[ModelCalibrator]] = {}
    
    @classmethod
    def register_calibrator(cls, 
                           calibrator_name: str, 
                           calibrator_class: Type[ModelCalibrator]) -> None:
        """
        Register a new model calibrator.
        
        Args:
            calibrator_name: Name identifier for the calibrator
            calibrator_class: Class implementing the calibrator
        """
        cls._calibrators[calibrator_name] = calibrator_class
        logger.info(f"Registered calibrator: {calibrator_name}")
    
    @classmethod
    def create_calibrator(cls, 
                         calibrator_name: str,
                         **kwargs) -> ModelCalibrator:
        """
        Create a calibrator instance.
        
        Args:
            calibrator_name: Name of the calibrator to create
            **kwargs: Additional arguments for calibrator construction
            
        Returns:
            Configured calibrator instance
            
        Raises:
            ValueError: If calibrator name is not registered
        """
        if calibrator_name not in cls._calibrators:
            available = list(cls._calibrators.keys())
            raise ValueError(f"Unknown calibrator: {calibrator_name}. "
                           f"Available calibrators: {available}")
        
        calibrator_class = cls._calibrators[calibrator_name]
        return calibrator_class(**kwargs)


def create_model_config_from_size(model_size: ModelSize) -> dict:
    """
    Create default model configuration based on model size.
    
    Args:
        model_size: Size of the model
        
    Returns:
        Dictionary with default configuration for the model size
    """
    size_configs = {
        ModelSize.SMALL: {
            "name": "Qwen-1.5B",
            "parameter_count": "1.5B",
            "expected_token_range": {
                "min_input_tokens": 10,
                "max_input_tokens": 2000,
                "min_output_tokens": 1,
                "max_output_tokens": 1000
            }
        },
        ModelSize.MEDIUM: {
            "name": "LLaMA-8B", 
            "parameter_count": "8B",
            "expected_token_range": {
                "min_input_tokens": 10,
                "max_input_tokens": 4000,
                "min_output_tokens": 1,
                "max_output_tokens": 2000
            }
        },
        ModelSize.LARGE: {
            "name": "Qwen-14B",
            "parameter_count": "14B", 
            "expected_token_range": {
                "min_input_tokens": 10,
                "max_input_tokens": 4000,
                "min_output_tokens": 1,
                "max_output_tokens": 2000
            }
        }
    }
    
    return size_configs[model_size]
