"""
Model calibration strategies for mapping server predictions to Jetson hardware.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from ..core.interfaces import ModelCalibrator, RegressionStrategy, PredictionResult, DataPreprocessor
from ..modeling.linear_regression import LinearRegressionStrategy
from ..modeling.polynomial_regression import PolynomialRegressionStrategy

logger = logging.getLogger(__name__)


class ModelCalibrator(ABC):
    """Abstract base class for model calibration strategies."""
    @abstractmethod
    def calibrate_model(
        self,
        server_model: RegressionStrategy,
        jetson_data: pd.DataFrame,
        preprocessor: DataPreprocessor,
    ) -> RegressionStrategy:
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
    def __init__(self, method: str = "mean_ratio", manual_scaling_factor: Optional[float] = None):
        self.method = method
        self.manual_scaling_factor = manual_scaling_factor
        self.scaling_factor: Optional[float] = None
        self.calibration_stats: Dict[str, Any] = {}

    def calibrate_model(
        self,
        server_model: RegressionStrategy,
        jetson_data: pd.DataFrame,
        preprocessor: DataPreprocessor,
    ) -> RegressionStrategy:
        logger.info(f"Calibrating server model using {self.method} method")
        if not server_model.is_fitted:
            raise ValueError("Server model must be fitted before calibration")
        jetson_features_df = preprocessor.preprocess_features(jetson_data)
        X_jetson, _ = preprocessor.create_feature_matrix(jetson_features_df)
        target_type = f"{preprocessor.phase}_latency"
        jetson_latency = preprocessor.extract_target_variable(jetson_data, target_type)
        server_result = server_model.predict(X_jetson)
        server_predictions = server_result.predictions
        self.scaling_factor = self.compute_scaling_factor(server_predictions, jetson_latency)
        self._compute_calibration_stats(server_predictions, jetson_latency)
        calibrated_model = ScaledRegressionWrapper(server_model, self.scaling_factor)
        logger.info(f"Calibration complete - scaling factor: {self.scaling_factor:.4f}")
        return calibrated_model

    def compute_scaling_factor(
        self, server_predictions: np.ndarray, jetson_measurements: np.ndarray
    ) -> float:
        if len(server_predictions) != len(jetson_measurements):
            raise ValueError("Prediction and measurement arrays must have same length")
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
            # Use sklearn LinearRegression with fit_intercept=False
            reg = LinearRegression(fit_intercept=False)
            reg.fit(server_valid.reshape(-1, 1), jetson_valid)
            scaling_factor = reg.coef_[0]
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        return float(scaling_factor)

    def _compute_calibration_stats(
        self, server_pred: np.ndarray, jetson_actual: np.ndarray
    ) -> None:
        if self.scaling_factor is None:
            return
        scaled_predictions = server_pred * self.scaling_factor
        mse = mean_squared_error(jetson_actual, scaled_predictions)
        mae = mean_absolute_error(jetson_actual, scaled_predictions)
        mape = mean_absolute_percentage_error(jetson_actual, scaled_predictions) * 100
        self.calibration_stats = {
            "scaling_factor": self.scaling_factor,
            "calibration_mse": float(mse),
            "calibration_mae": float(mae),
            "calibration_mape": float(mape),
            "n_calibration_samples": len(jetson_actual),
            "method": self.method,
        }


class DirectJetsonCalibrator(ModelCalibrator):
    """Direct calibration by training a separate model on Jetson data."""
    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type
        self.jetson_model: Optional[RegressionStrategy] = None
        self.calibration_stats: Dict[str, Any] = {}

    def calibrate_model(
        self,
        server_model: RegressionStrategy,
        jetson_data: pd.DataFrame,
        preprocessor: DataPreprocessor,
    ) -> RegressionStrategy:
        logger.info(f"Training direct Jetson model using {self.model_type}")
        jetson_features_df = preprocessor.preprocess_features(jetson_data)
        X_jetson, _ = preprocessor.create_feature_matrix(jetson_features_df)
        target_type = f"{preprocessor.phase}_latency"
        jetson_latency = preprocessor.extract_target_variable(jetson_data, target_type)
        if self.model_type == "linear":
            self.jetson_model = LinearRegressionStrategy()
        elif self.model_type == "polynomial":
            degree = getattr(server_model, "degree", 2)
            self.jetson_model = PolynomialRegressionStrategy(degree=degree)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.jetson_model.fit(X_jetson, jetson_latency)
        self._compute_calibration_stats(X_jetson, jetson_latency, server_model)
        logger.info("Direct Jetson model training complete")
        return self.jetson_model

    def compute_scaling_factor(
        self, server_predictions: np.ndarray, jetson_measurements: np.ndarray
    ) -> float:
        return 1.0

    def _compute_calibration_stats(
        self, X: np.ndarray, jetson_latency: np.ndarray, server_model: RegressionStrategy
    ) -> None:
        jetson_pred = self.jetson_model.predict(X)
        jetson_mse = mean_squared_error(jetson_latency, jetson_pred.predictions)
        server_pred = server_model.predict(X)
        server_mse = mean_squared_error(jetson_latency, server_pred.predictions)
        self.calibration_stats = {
            "jetson_model_mse": float(jetson_mse),
            "server_model_mse": float(server_mse),
            "improvement_ratio": float(server_mse / jetson_mse) if jetson_mse > 0 else float("inf"),
            "n_calibration_samples": len(jetson_latency),
            "method": f"direct_{self.model_type}",
        }


class ScaledRegressionWrapper(RegressionStrategy):
    """
    Wraps a regression model and applies a scaling factor to its predictions.
    """
    def __init__(self, base_model: RegressionStrategy, scaling_factor: float):
        self.base_model = base_model
        self.scaling_factor = scaling_factor
        self.is_fitted = base_model.is_fitted
        self._feature_names = base_model.feature_names

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.warning("ScaledRegressionWrapper does not fit, fitting is done on the base model.")
        pass

    def predict(self, X: np.ndarray) -> PredictionResult:
        base_prediction_result = self.base_model.predict(X)
        base_predictions = base_prediction_result.predictions
        scaled_predictions = base_predictions * self.scaling_factor
        confidence_intervals = None
        if base_prediction_result.confidence_intervals is not None:
            lower_bound, upper_bound = base_prediction_result.confidence_intervals
            confidence_intervals = (
                lower_bound * self.scaling_factor,
                upper_bound * self.scaling_factor,
            )
        return PredictionResult(predictions=scaled_predictions, confidence_intervals=confidence_intervals)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return self.base_model.get_feature_importance()

    def get_hyperparameters(self) -> Dict[str, Any]:
        return self.base_model.get_hyperparameters()

    def set_feature_names(self, feature_names: List[str]) -> None:
        self._feature_names = feature_names
        self.base_model.set_feature_names(feature_names)

    @property
    def feature_names(self) -> Optional[List[str]]:
        return self._feature_names


def create_calibrator(calibration_method: str, preprocessor: DataPreprocessor, manual_scaling_factor: Optional[float] = None) -> ModelCalibrator:
    """
    Factory function to create a ModelCalibrator instance.
    """
    if calibration_method == "simple_scaling":
        return SimpleScalingCalibrator(method="mean_ratio", manual_scaling_factor=manual_scaling_factor)
    else:
        raise ValueError(f"Unknown calibration method: {calibration_method}")
