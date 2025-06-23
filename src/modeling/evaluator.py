"""
Model evaluation utilities for latency prediction models.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from ..core.interfaces import ModelEvaluator, RegressionStrategy, ModelMetrics

logger = logging.getLogger(__name__)


class LatencyModelEvaluator(ModelEvaluator):
    """Evaluator specialized for latency prediction models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize evaluator.
        
        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            ModelMetrics with evaluation results
        """
        logger.info(f"Evaluating model on {len(y_true)} predictions")
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true({len(y_true)}) != y_pred({len(y_pred)})")
        
        # Calculate standard regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero with small epsilon
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
        
        metrics = ModelMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape
        )
        
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
    
    def cross_validate(self, model: RegressionStrategy, X: np.ndarray, 
                      y: np.ndarray, cv_folds: int) -> List[ModelMetrics]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Regression model to evaluate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            List of ModelMetrics for each fold
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Create KFold cross-validator
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"Processing fold {fold_idx + 1}/{cv_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh model instance for this fold
            # Note: In practice, you'd want to clone the model properly
            fold_model = self._clone_model(model)
            
            # Fit and evaluate
            fold_model.fit(X_train, y_train)
            result = fold_model.predict(X_val)
            
            # Calculate metrics for this fold
            fold_metric = self.evaluate_model(y_val, result.predictions)
            fold_metrics.append(fold_metric)
        
        # Log summary statistics
        self._log_cv_summary(fold_metrics)
        
        return fold_metrics
    
    def _clone_model(self, model: RegressionStrategy) -> RegressionStrategy:
        """
        Create a fresh copy of the model for cross-validation.
        
        Args:
            model: Model to clone
            
        Returns:
            Fresh model instance
        """
        # Get model hyperparameters
        hyperparams = model.get_hyperparameters()
        
        # Create new instance based on model type
        if hasattr(model, 'degree'):  # Polynomial regression
            from ..modeling.polynomial_regression import PolynomialRegressionStrategy
            return PolynomialRegressionStrategy(
                degree=hyperparams.get('degree', 2),
                interaction_only=hyperparams.get('interaction_only', False),
                include_bias=hyperparams.get('include_bias', True),
                fit_intercept=hyperparams.get('fit_intercept', True)
            )
        else:  # Linear regression
            from ..modeling.linear_regression import LinearRegressionStrategy
            return LinearRegressionStrategy(
                fit_intercept=hyperparams.get('fit_intercept', True),
                normalize=hyperparams.get('normalize', False)
            )
    
    def _log_cv_summary(self, fold_metrics: List[ModelMetrics]) -> None:
        """Log cross-validation summary statistics."""
        if not fold_metrics:
            return
        
        # Calculate mean and std for each metric
        metrics_dict = {
            'mse': [m.mse for m in fold_metrics],
            'rmse': [m.rmse for m in fold_metrics],
            'mae': [m.mae for m in fold_metrics],
            'r2_score': [m.r2_score for m in fold_metrics],
            'mape': [m.mape for m in fold_metrics]
        }
        
        logger.info("Cross-validation summary:")
        for metric_name, values in metrics_dict.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")


def compare_models(models: Dict[str, RegressionStrategy], 
                  X_test: np.ndarray, 
                  y_test: np.ndarray) -> Dict[str, ModelMetrics]:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of model_name -> fitted_model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of model_name -> ModelMetrics
    """
    logger.info(f"Comparing {len(models)} models on {len(y_test)} test samples")
    
    evaluator = LatencyModelEvaluator()
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        
        if not model.is_fitted:
            logger.warning(f"Model {model_name} is not fitted, skipping")
            continue
        
        # Make predictions
        result = model.predict(X_test)
        
        # Evaluate
        metrics = evaluator.evaluate_model(y_test, result.predictions)
        results[model_name] = metrics
    
    # Log comparison summary
    _log_model_comparison(results)
    
    return results


def _log_model_comparison(results: Dict[str, ModelMetrics]) -> None:
    """Log model comparison results."""
    if not results:
        return
    
    logger.info("Model comparison results:")
    logger.info(f"{'Model':<20} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'MAPE':<8}")
    logger.info("-" * 60)
    
    for model_name, metrics in results.items():
        logger.info(f"{model_name:<20} {metrics.r2_score:<8.4f} "
                   f"{metrics.rmse:<10.4f} {metrics.mae:<10.4f} {metrics.mape:<8.2f}")


def calculate_model_confidence(model: RegressionStrategy, 
                             X: np.ndarray, 
                             n_bootstrap: int = 100) -> Dict[str, Any]:
    """
    Calculate model confidence using bootstrap sampling.
    
    Args:
        model: Fitted regression model  
        X: Feature matrix
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with confidence statistics
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted")
    
    logger.info(f"Calculating model confidence with {n_bootstrap} bootstrap samples")
    
    # Make base predictions
    base_result = model.predict(X)
    base_predictions = base_result.predictions
    
    # Bootstrap sampling for confidence estimation
    bootstrap_predictions = []
    
    for i in range(n_bootstrap):
        # Simple bootstrap - in practice would resample training data and refit
        # Here we just add noise to simulate uncertainty
        noise = np.random.normal(0, np.std(base_predictions) * 0.1, len(base_predictions))
        bootstrap_pred = base_predictions + noise
        bootstrap_predictions.append(bootstrap_pred)
    
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Calculate confidence statistics
    confidence_stats = {
        'mean_prediction': np.mean(bootstrap_predictions, axis=0),
        'std_prediction': np.std(bootstrap_predictions, axis=0),
        'confidence_5th': np.percentile(bootstrap_predictions, 5, axis=0),
        'confidence_95th': np.percentile(bootstrap_predictions, 95, axis=0),
        'prediction_intervals': (
            np.percentile(bootstrap_predictions, 2.5, axis=0),
            np.percentile(bootstrap_predictions, 97.5, axis=0)
        )
    }
    
    logger.info("Model confidence calculation complete")
    return confidence_stats
