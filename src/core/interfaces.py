"""
Core interfaces and abstract base classes for the modeling framework.

This module defines the abstract interfaces that implement the Strategy pattern
for different regression backends and the Template Method pattern for
the overall modeling workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..core.config import ExperimentConfig


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    
    mse: float
    rmse: float
    mae: float
    r2_score: float
    mape: float
    
    def __str__(self) -> str:
        return (f"MSE: {self.mse:.4f}, RMSE: {self.rmse:.4f}, "
                f"MAE: {self.mae:.4f}, R²: {self.r2_score:.4f}, MAPE: {self.mape:.2f}%")


@dataclass
class PredictionResult:
    """Container for prediction results with confidence intervals."""
    
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    feature_importance: Optional[Dict[str, float]] = None


class RegressionStrategy(ABC):
    """Abstract base class for regression strategies."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            PredictionResult containing predictions and optional metadata
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if supported by the model.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get current hyperparameters of the model.
        
        Returns:
            Dictionary of hyperparameter names and values
        """
        pass


class DataLoader(ABC):
    """Abstract base class for data loading strategies."""
    
    @abstractmethod
    def load_server_data(self, model_name: str) -> pd.DataFrame:
        """
        Load server-side training data for a specific model.
        
        Args:
            model_name: Name of the model to load data for
            
        Returns:
            DataFrame with server measurements
        """
        pass
    
    @abstractmethod
    def load_jetson_data(self, model_name: str) -> pd.DataFrame:
        """
        Load Jetson calibration data for a specific model.
        
        Args:
            model_name: Name of the model to load data for
            
        Returns:
            DataFrame with Jetson measurements
        """
        pass
    
    @abstractmethod
    def validate_data_schema(self, df: pd.DataFrame, data_type: str) -> bool:
        """
        Validate that DataFrame has expected schema.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data ("server" or "jetson")
            
        Returns:
            True if schema is valid
            
        Raises:
            ValueError: If schema validation fails
        """
        pass


class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing strategies."""
    
    @abstractmethod
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data into model features.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame with features
        """
        pass
    
    @abstractmethod
    def extract_target_variable(self, df: pd.DataFrame, target_type: str) -> np.ndarray:
        """
        Extract target variable from DataFrame.
        
        Args:
            df: Preprocessed DataFrame
            target_type: Type of target ("latency", "energy", "power")
            
        Returns:
            Target variable array
        """
        pass
    
    @abstractmethod
    def create_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix from preprocessed DataFrame.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        pass


class ModelCalibrator(ABC):
    """Abstract base class for hardware-specific model calibration."""
    
    @abstractmethod
    def calibrate_model(self, 
                       server_model: RegressionStrategy,
                       jetson_data: pd.DataFrame) -> RegressionStrategy:
        """
        Calibrate server model for Jetson hardware.
        
        Args:
            server_model: Trained server-side model
            jetson_data: Jetson calibration data
            
        Returns:
            Calibrated model for Jetson predictions
        """
        pass
    
    @abstractmethod
    def compute_scaling_factor(self, 
                              server_predictions: np.ndarray,
                              jetson_measurements: np.ndarray) -> float:
        """
        Compute scaling factor between server and Jetson measurements.
        
        Args:
            server_predictions: Server model predictions
            jetson_measurements: Actual Jetson measurements
            
        Returns:
            Scaling factor for calibration
        """
        pass


class ModelEvaluator(ABC):
    """Abstract base class for model evaluation."""
    
    @abstractmethod
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray) -> ModelMetrics:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            ModelMetrics object with evaluation results
        """
        pass
    
    @abstractmethod
    def cross_validate(self, 
                      model: RegressionStrategy,
                      X: np.ndarray, 
                      y: np.ndarray,
                      cv_folds: int) -> List[ModelMetrics]:
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
        pass


class LatencyPredictor(ABC):
    """
    Template method interface for the complete latency prediction workflow.
    
    This class defines the overall algorithm structure while allowing
    subclasses to customize specific steps.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._server_model: Optional[RegressionStrategy] = None
        self._jetson_model: Optional[RegressionStrategy] = None
    
    def predict_latency(self, 
                       input_tokens: int, 
                       output_tokens: int) -> Dict[str, float]:
        """
        Main template method for latency prediction.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with prediction results
        """
        self._validate_inputs(input_tokens, output_tokens)
        
        if not self._jetson_model:
            raise RuntimeError("Model must be trained before making predictions")
        
        features = self._prepare_features(input_tokens, output_tokens)
        result = self._jetson_model.predict(features)
        
        return self._format_prediction_output(result, input_tokens, output_tokens)
    
    @abstractmethod
    def train_models(self) -> None:
        """Train both server and Jetson models."""
        pass
    
    @abstractmethod
    def _validate_inputs(self, input_tokens: int, output_tokens: int) -> None:
        """Validate input parameters."""
        pass
    
    @abstractmethod
    def _prepare_features(self, input_tokens: int, output_tokens: int) -> np.ndarray:
        """Prepare feature vector for prediction."""
        pass
    
    @abstractmethod
    def _format_prediction_output(self, 
                                 result: PredictionResult,
                                 input_tokens: int, 
                                 output_tokens: int) -> Dict[str, float]:
        """Format prediction results for output."""
        pass
