"""
Polynomial regression strategy implementation for latency prediction.
"""

import numpy as np
from typing import Dict, Optional, Any
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from ..core.interfaces import RegressionStrategy, PredictionResult, ModelMetrics

logger = logging.getLogger(__name__)


class PolynomialRegressionStrategy(RegressionStrategy):
    """Polynomial regression implementation for token-based latency prediction."""
    
    def __init__(self, degree: int = 2, interaction_only: bool = False, 
                 include_bias: bool = True, fit_intercept: bool = True):
        """
        Initialize polynomial regression strategy.
        
        Args:
            degree: Degree of polynomial features
            interaction_only: If True, only interaction features are produced
            include_bias: If True, include bias column
            fit_intercept: Whether to fit intercept in linear regression
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.fit_intercept = fit_intercept
        
        # Create pipeline with polynomial features and linear regression
        self.model = Pipeline([
            ('poly', PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=include_bias
            )),
            ('linear', LinearRegression(fit_intercept=fit_intercept))
        ])
        
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        self.poly_feature_names: Optional[list] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit polynomial regression model to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        logger.info(f"Fitting polynomial regression (degree={self.degree}) on "
                   f"{X.shape[0]} samples, {X.shape[1]} features")
        
        # Validate inputs
        if X.shape[0] != len(y):
            raise ValueError(f"X samples ({X.shape[0]}) != y samples ({len(y)})")
        
        if X.shape[0] < self.degree + 1:
            raise ValueError(f"Need at least {self.degree + 1} samples for degree {self.degree}")
        
        # Fit pipeline
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Get polynomial feature names for interpretability
        poly_transformer = self.model.named_steps['poly']
        if self.feature_names:
            self.poly_feature_names = poly_transformer.get_feature_names_out(self.feature_names)
        else:
            input_features = [f"x{i}" for i in range(X.shape[1])]
            self.poly_feature_names = poly_transformer.get_feature_names_out(input_features)
        
        # Log model performance on training data
        train_pred = self.model.predict(X)
        train_r2 = r2_score(y, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        
        logger.info(f"Polynomial regression fitted - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        logger.info(f"Generated {len(self.poly_feature_names)} polynomial features")
    
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
        
        logger.info(f"Making polynomial predictions for {X.shape[0]} samples")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate prediction intervals (approximate)
        confidence_intervals = self._calculate_prediction_intervals(X, predictions)
        
        # Get feature importance (coefficients of polynomial features)
        feature_importance = self._get_coefficient_importance()
        
        return PredictionResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            feature_importance=feature_importance
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance based on polynomial coefficients."""
        return self._get_coefficient_importance()
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        hyperparams = {
            "degree": self.degree,
            "interaction_only": self.interaction_only,
            "include_bias": self.include_bias,
            "fit_intercept": self.fit_intercept
        }
        
        if self.is_fitted:
            linear_model = self.model.named_steps['linear']
            hyperparams.update({
                "n_original_features": self.model.named_steps['poly'].n_features_in_,
                "n_poly_features": len(self.poly_feature_names) if self.poly_feature_names else None,
                "intercept": float(linear_model.intercept_)
            })
        
        return hyperparams
    
    def _calculate_prediction_intervals(self, X: np.ndarray, 
                                     predictions: np.ndarray) -> tuple:
        """
        Calculate approximate prediction intervals.
        
        Args:
            X: Feature matrix
            predictions: Model predictions
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        # Simple approximation - in practice would use proper statistical methods
        pred_std = np.std(predictions)
        margin = 1.96 * pred_std  # 95% confidence interval approximation
        
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
        
        return (lower_bounds, upper_bounds)
    
    def _get_coefficient_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance based on absolute coefficient values."""
        if not self.is_fitted or not self.poly_feature_names:
            return None
        
        linear_model = self.model.named_steps['linear']
        coefficients = linear_model.coef_
        
        # Map polynomial features to importance
        importance = {
            name: abs(coef) for name, coef in zip(self.poly_feature_names, coefficients)
        }
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        # Group by original features for better interpretability
        grouped_importance = self._group_polynomial_importance(importance)
        
        return grouped_importance
    
    def _group_polynomial_importance(self, poly_importance: Dict[str, float]) -> Dict[str, float]:
        """Group polynomial feature importance by original features."""
        if not self.feature_names:
            return poly_importance
        
        grouped = {}
        
        # Initialize with original feature names
        for feature_name in self.feature_names:
            grouped[feature_name] = 0.0
        
        # Sum importance for features containing each original feature
        for poly_feature, importance in poly_importance.items():
            for original_feature in self.feature_names:
                if original_feature in poly_feature:
                    grouped[original_feature] += importance
        
        return grouped
    
    def set_feature_names(self, feature_names: list) -> None:
        """Set feature names for interpretability."""
        self.feature_names = feature_names


def create_polynomial_latency_model(degree: int = 2, 
                                  interaction_only: bool = False) -> PolynomialRegressionStrategy:
    """
    Convenience function to create polynomial regression model for latency prediction.
    
    Args:
        degree: Polynomial degree
        interaction_only: Whether to include only interaction terms
        
    Returns:
        Configured PolynomialRegressionStrategy
    """
    model = PolynomialRegressionStrategy(
        degree=degree,
        interaction_only=interaction_only
    )
    logger.info(f"Created polynomial regression strategy (degree={degree}) for latency prediction")
    return model


def evaluate_polynomial_model(model: PolynomialRegressionStrategy,
                            X_test: np.ndarray, 
                            y_test: np.ndarray) -> ModelMetrics:
    """
    Evaluate polynomial regression model performance.
    
    Args:
        model: Fitted polynomial regression model
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
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
    
    metrics = ModelMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2_score=r2,
        mape=mape
    )
    
    logger.info(f"Polynomial model evaluation: {metrics}")
    return metrics
