"""
Registration and factory setup for regression strategies.
"""

from ..core.factory import RegressionFactory
from ..core.config import RegressionType
from ..modeling.linear_regression import LinearRegressionStrategy
from ..modeling.polynomial_regression import PolynomialRegressionStrategy


def register_regression_strategies() -> None:
    """Register all regression strategies with the factory."""
    
    # Register linear regression
    RegressionFactory.register_strategy(RegressionType.LINEAR, LinearRegressionStrategy)
    
    # Register polynomial regression  
    RegressionFactory.register_strategy(RegressionType.POLYNOMIAL, PolynomialRegressionStrategy)


def create_latency_models(config_dict: dict) -> dict:
    """
    Create configured latency prediction models.
    
    Args:
        config_dict: Configuration dictionary with hyperparameters
        
    Returns:
        Dictionary of model_name -> configured_model
    """
    register_regression_strategies()
    
    models = {}
    
    # Linear regression model
    if 'linear' in config_dict:
        linear_config = config_dict['linear']
        models['linear'] = RegressionFactory.create_strategy(
            RegressionType.LINEAR,
            **linear_config
        )
    
    # Polynomial regression model
    if 'polynomial' in config_dict:
        poly_config = config_dict['polynomial']
        models['polynomial'] = RegressionFactory.create_strategy(
            RegressionType.POLYNOMIAL,
            **poly_config
        )
    
    return models


def get_default_latency_models() -> dict:
    """
    Create default latency prediction models with standard configurations.
    
    Returns:
        Dictionary of configured models
    """
    default_config = {
        'linear': {
            'fit_intercept': True,
            'normalize': False
        },
        'polynomial': {
            'degree': 2,
            'interaction_only': False,
            'include_bias': True,
            'fit_intercept': True
        }
    }
    
    return create_latency_models(default_config)


# Convenience function for quick setup
def quick_regression_setup() -> dict:
    """Quick setup with default regression models."""
    return get_default_latency_models()
