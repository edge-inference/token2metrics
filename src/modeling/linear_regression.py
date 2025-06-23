"""
Linear regression strategy implementation for latency prediction.
"""

import numpy as np
from typing import Dict, Optional, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from ..core.interfaces import RegressionStrategy, PredictionResult, ModelMetrics

logger = logging.getLogger(__name__)


class LinearRegressionStrategy(RegressionStrategy):
    """Linear regression implementation for token-based latency prediction."""
    
    def __init__(self, fit_intercept: bool = True, normalize: bool = False):
        """
        Initialize linear regression strategy.
        
        Args:
            fit_intercept: Whether to fit intercept term
            normalize: Whether to normalize features (deprecated in sklearn)
        """
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.is_fitted = False
        self.feature_names: Optional[list] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit linear regression model to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        logger.info(f"Fitting linear regression on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Validate inputs
        if X.shape[0] != len(y):
            raise ValueError(f"X samples ({X.shape[0]}) != y samples ({len(y)})")
        
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for regression")
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Log model performance on training data
        train_pred = self.model.predict(X)
        train_r2 = r2_score(y, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        
        logger.info(f"Linear regression fitted - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            PredictionResult with predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        logger.info(f"Making predictions for {X.shape[0]} samples")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate prediction intervals (approximate for linear regression)
        confidence_intervals = self._calculate_prediction_intervals(X, predictions)
        
        # Get feature importance (coefficients)
        feature_importance = self._get_coefficient_importance()
        
        return PredictionResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            feature_importance=feature_importance
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance based on coefficients."""
        return self._get_coefficient_importance()
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "n_features": self.model.n_features_in_ if self.is_fitted else None,
            "intercept": float(self.model.intercept_) if self.is_fitted else None
        }
    
    def _calculate_prediction_intervals(self, X: np.ndarray, 
                                     predictions: np.ndarray) -> tuple:
        """
        Calculate approximate prediction intervals for linear regression.
        
        Args:
            X: Feature matrix
            predictions: Model predictions
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        # Simple approximation based on prediction variance
        # In practice, would use proper statistical methods
        pred_std = np.std(predictions)
        margin = 1.96 * pred_std  # 95% confidence interval approximation
        
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
        
        return (lower_bounds, upper_bounds)
    
    def _get_coefficient_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance based on absolute coefficient values."""
        if not self.is_fitted:
            return None
        
        coefficients = self.model.coef_
        
        # If we have feature names, use them
        if self.feature_names:
            importance = {
                name: abs(coef) for name, coef in zip(self.feature_names, coefficients)
            }
        else:
            # Use generic feature names
            importance = {
                f"feature_{i}": abs(coef) for i, coef in enumerate(coefficients)
            }
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def set_feature_names(self, feature_names: list) -> None:
        """Set feature names for interpretability."""
        self.feature_names = feature_names


def create_linear_latency_model(fit_intercept: bool = True) -> LinearRegressionStrategy:
    """
    Convenience function to create a linear regression model for latency prediction.
    
    Args:
        fit_intercept: Whether to fit intercept
        
    Returns:
        Configured LinearRegressionStrategy
    """
    model = LinearRegressionStrategy(fit_intercept=fit_intercept)
    logger.info("Created linear regression strategy for latency prediction")
    return model


def evaluate_linear_model(model: LinearRegressionStrategy, 
                         X_test: np.ndarray, 
                         y_test: np.ndarray) -> ModelMetrics:
    """
    Evaluate linear regression model performance.
    
    Args:
        model: Fitted linear regression model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        ModelMetrics with evaluation results
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted before evaluation")
    
    # Make predictions
    result = model.predict(X_test)
    y_pred = result.predictions
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
    
    metrics = ModelMetrics(
        mse=mse,
        rmse=rmse,  
        mae=mae,
        r2_score=r2,
        mape=mape
    )
    
    logger.info(f"Linear model evaluation: {metrics}")
    return metrics
