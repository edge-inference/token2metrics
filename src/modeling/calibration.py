"""
Model calibration strategies for mapping server predictions to Jetson hardware.
"""

import logging

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List # Added List
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod # Added ABC, abstractmethod

from ..core.interfaces import ModelCalibrator, RegressionStrategy, PredictionResult, DataPreprocessor # Added DataPreprocessor
from ..modeling.linear_regression import LinearRegressionStrategy
from ..modeling.polynomial_regression import PolynomialRegressionStrategy

logger = logging.getLogger(__name__)


class ModelCalibrator(ABC):
    """Abstract base class for model calibration strategies."""
    
    @abstractmethod
    def calibrate_model(self, server_model: RegressionStrategy, 
                       jetson_data: pd.DataFrame, 
                       preprocessor: DataPreprocessor) -> RegressionStrategy: # Corrected type hint
        """
        Calibrate a server-side model for Jetson predictions.
        
        Args:
            server_model: Trained server-side model
            jetson_data: Raw Jetson data DataFrame
            preprocessor: Data preprocessor instance
            
        Returns:
            Calibrated model wrapped with calibration logic
        """
        pass


class SimpleScalingCalibrator(ModelCalibrator):
    """Simple scaling factor calibration between server and Jetson."""
    
    def __init__(self, method: str = "mean_ratio"):
        """
        Initialize scaling calibrator.
        
        Args:
            method: Scaling method ("mean_ratio", "median_ratio", "linear_fit")
        """
        self.method = method
        self.scaling_factor: Optional[float] = None
        self.calibration_stats: Dict[str, Any] = {}
    
    def calibrate_model(self, server_model: RegressionStrategy, 
                       jetson_data: pd.DataFrame, 
                       preprocessor: DataPreprocessor) -> RegressionStrategy: # Corrected type hint
        """
        Calibrate server model for Jetson predictions using scaling.
        
        Args:
            server_model: Trained server-side model
            jetson_data: Array with columns [tokens, jetson_latency]
            preprocessor: Data preprocessor instance
            
        Returns:
            Calibrated model wrapped with scaling
        """
        logger.info(f"Calibrating server model using {self.method} method")
        
        if not server_model.is_fitted:
            raise ValueError("Server model must be fitted before calibration")
        
        # Use the preprocessor to prepare features from Jetson data
        jetson_features_df = preprocessor.preprocess_features(jetson_data)
        X_jetson, _ = preprocessor.create_feature_matrix(jetson_features_df)
        
        # Extract actual Jetson latency based on the phase the preprocessor is configured for
        target_type = f"{preprocessor.phase}_latency"
        jetson_latency = preprocessor.extract_target_variable(jetson_data, target_type)
        
        # Get server predictions for the same token counts (using the prepared Jetson features)
        server_result = server_model.predict(X_jetson)
        server_predictions = server_result.predictions
        
        # Compute scaling factor
        self.scaling_factor = self.compute_scaling_factor(
            server_predictions, jetson_latency
        )
        
        # Store calibration statistics
        self._compute_calibration_stats(server_predictions, jetson_latency)
        
        # Create calibrated model wrapper
        calibrated_model = ScaledRegressionWrapper(server_model, self.scaling_factor)
        
        logger.info(f"Calibration complete - scaling factor: {self.scaling_factor:.4f}")
        return calibrated_model
    
    def compute_scaling_factor(self, server_predictions: np.ndarray,
                              jetson_measurements: np.ndarray) -> float:
        """
        Compute scaling factor between server and Jetson measurements.
        
        Args:
            server_predictions: Server model predictions
            jetson_measurements: Actual Jetson measurements
            
        Returns:   
            Scaling factor
        """
        if len(server_predictions) != len(jetson_measurements):
            raise ValueError("Prediction and measurement arrays must have same length")
        
        # Avoid division by zero
        valid_idx = server_predictions != 0
        if not valid_idx.any():
            raise ValueError("All server predictions are zero")
        
        server_valid = server_predictions[valid_idx]
        jetson_valid = jetson_measurements[valid_idx]
        
        if self.method == "mean_ratio":
            scaling_factor = np.mean(jetson_valid / server_valid)
        elif self.method == "median_ratio":
            scaling_factor = np.median(jetson_valid / server_valid)
        elif self.method == "linear_fit":
            # Fit y = ax where y=jetson, x=server (force through origin)
            scaling_factor = np.sum(jetson_valid * server_valid) / np.sum(server_valid ** 2)
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        return float(scaling_factor)
    
    def _compute_calibration_stats(self, server_pred: np.ndarray, 
                                  jetson_actual: np.ndarray) -> None:
        """Compute calibration quality statistics."""
        if self.scaling_factor is None:
            return
        
        # Apply scaling to server predictions
        scaled_predictions = server_pred * self.scaling_factor
        
        # Calculate calibration metrics
        mse = np.mean((jetson_actual - scaled_predictions) ** 2)
        mae = np.mean(np.abs(jetson_actual - scaled_predictions))
        mape = np.mean(np.abs((jetson_actual - scaled_predictions) / 
                            np.maximum(jetson_actual, 1e-8))) * 100
        
        self.calibration_stats = {
            "scaling_factor": self.scaling_factor,
            "calibration_mse": float(mse),
            "calibration_mae": float(mae), 
            "calibration_mape": float(mape),
            "n_calibration_samples": len(jetson_actual),
            "method": self.method
        }


class DirectJetsonCalibrator(ModelCalibrator):
    """Direct calibration by training a separate model on Jetson data."""
    
    def __init__(self, model_type: str = "linear"):
        """
        Initialize direct calibrator.
        
        Args:
            model_type: Type of model to train on Jetson data
        """
        self.model_type = model_type
        self.jetson_model: Optional[RegressionStrategy] = None
        self.calibration_stats: Dict[str, Any] = {}
    
    def calibrate_model(self, server_model: RegressionStrategy,
                       jetson_data: np.ndarray) -> RegressionStrategy:
        """
        Train a separate model directly on Jetson data.
        
        Args:
            server_model: Server model (used for initialization/comparison)
            jetson_data: Array with columns [tokens, jetson_latency]
            
        Returns:
            New model trained on Jetson data
        """
        logger.info(f"Training direct Jetson model using {self.model_type}")
        
        # Extract features and targets
        tokens = jetson_data[:, 0].reshape(-1, 1)
        jetson_latency = jetson_data[:, 1]
        
        # Create new model for Jetson data
        if self.model_type == "linear":
            self.jetson_model = LinearRegressionStrategy()
        elif self.model_type == "polynomial":
            # Use same degree as server model if polynomial
            degree = getattr(server_model, 'degree', 2)
            self.jetson_model = PolynomialRegressionStrategy(degree=degree)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train on Jetson data
        self.jetson_model.fit(tokens, jetson_latency)
        
        # Compute calibration statistics
        self._compute_calibration_stats(tokens, jetson_latency, server_model)
        
        logger.info("Direct Jetson model training complete")
        return self.jetson_model
    
    def compute_scaling_factor(self, server_predictions: np.ndarray,
                              jetson_measurements: np.ndarray) -> float:
        """Not used in direct calibration, but required by interface."""
        return 1.0
    
    def _compute_calibration_stats(self, tokens: np.ndarray, jetson_latency: np.ndarray,
                                  server_model: RegressionStrategy) -> None:
        """Compute statistics comparing direct Jetson model to server model."""
        # Jetson model performance
        jetson_pred = self.jetson_model.predict(tokens)
        jetson_mse = np.mean((jetson_latency - jetson_pred.predictions) ** 2)
        
        # Server model performance on same data (for comparison)
        server_pred = server_model.predict(tokens)
        server_mse = np.mean((jetson_latency - server_pred.predictions) ** 2)
        
        self.calibration_stats = {
            "jetson_model_mse": float(jetson_mse),
            "server_model_mse": float(server_mse),
            "improvement_ratio": float(server_mse / jetson_mse) if jetson_mse > 0 else float('inf'),
            "n_calibration_samples": len(jetson_latency),
            "method": f"direct_{self.model_type}"
        }


class ScaledRegressionWrapper(RegressionStrategy):
    """
    Wraps a regression model and applies a scaling factor to its predictions.
    """
    
    def __init__(self, base_model: RegressionStrategy, scaling_factor: float):
        """
        Initialize scaled wrapper.
        
        Args:
            base_model: The base regression model.
            scaling_factor: The scaling factor to apply.
        """
        self.base_model = base_model
        self.scaling_factor = scaling_factor
        self.is_fitted = base_model.is_fitted # Inherit fitted state
        self._feature_names = base_model.feature_names # Inherit feature names

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fitting is done on the base model, this method does nothing.
        """
        logger.warning("ScaledRegressionWrapper does not fit, fitting is done on the base model.")
        pass # Fitting is done on the base model

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Make predictions using the base model and apply the scaling factor.
        """
        base_predictions = self.base_model.predict(X).predictions
        scaled_predictions = base_predictions * self.scaling_factor
        
        # If base model provides confidence intervals, scale them too
        confidence_intervals = None
        # Check if base_model.predict(X) returns confidence_intervals before accessing
        base_prediction_result = self.base_model.predict(X)
        if base_prediction_result.confidence_intervals is not None:
             lower_bound, upper_bound = base_prediction_result.confidence_intervals
             confidence_intervals = (lower_bound * self.scaling_factor, upper_bound * self.scaling_factor)

        return PredictionResult(predictions=scaled_predictions, confidence_intervals=confidence_intervals)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the base model.
        """
        return self.base_model.get_feature_importance()

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get hyperparameters from the base model.
        """
        return self.base_model.get_hyperparameters()

    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set feature names for the wrapper and the base model.
        """
        self._feature_names = feature_names
        self.base_model.set_feature_names(feature_names)

    @property
    def feature_names(self) -> Optional[List[str]]:
        """
        Get the feature names.
        """
        return self._feature_names


def create_calibrator(calibration_method: str, preprocessor: DataPreprocessor) -> ModelCalibrator: # Corrected type hint
    """
    Factory function to create a ModelCalibrator instance.
    
    Args:
        calibration_method: The calibration method to use.
        preprocessor: The data preprocessor instance.
        
    Returns:
        A ModelCalibrator instance.
    """
    if calibration_method == "simple_scaling":
        return SimpleScalingCalibrator(method="mean_ratio") # Default to mean_ratio for now
    else:
        raise ValueError(f"Unknown calibration method: {calibration_method}")
