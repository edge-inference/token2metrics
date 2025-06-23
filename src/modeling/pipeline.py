"""
Complete training pipeline for phase-specific latency predictors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from ..core.interfaces import LatencyPredictor, RegressionStrategy, PredictionResult
from ..core.config import ExperimentConfig, ModelSize
from ..data.registry import create_default_data_setup
from ..modeling.registry import get_default_latency_models
from ..modeling.calibration import create_calibrator
from ..modeling.evaluator import LatencyModelEvaluator
from ..processors.token_preprocessor import PrefillDecodePreprocessor
from ..utils.helpers import ExperimentTracker, save_experiment_results

logger = logging.getLogger(__name__)


class PhaseSpecificLatencyPredictor(LatencyPredictor):
    """Predictor for either prefill or decode phase latency."""
    
    def __init__(self, config: ExperimentConfig, phase: str):
        """
        Initialize phase-specific predictor.
        
        Args:
            config: Experiment configuration
            phase: "prefill" or "decode"
        """
        super().__init__(config)
        
        if phase not in ["prefill", "decode"]:
            raise ValueError(f"Phase must be 'prefill' or 'decode', got {phase}")
        
        self.phase = phase
        self.preprocessor = PrefillDecodePreprocessor(phase=phase)
        self.evaluator = LatencyModelEvaluator()
        
        # Model components
        self._server_model: Optional[RegressionStrategy] = None
        self._jetson_model: Optional[RegressionStrategy] = None
        self._calibrator = None
        
        # Data components
        self._data_setup = None
        self._server_data: Optional[pd.DataFrame] = None
        self._jetson_data: Optional[pd.DataFrame] = None
    
    def train_models(self) -> None:
        """Train both server and calibrated Jetson models."""
        logger.info(f"Starting {self.phase} model training for {self.config.model_config.name}")
        
        # Setup data loading
        self._setup_data_loading()
        
        # Load and preprocess data
        self._load_and_preprocess_data()
        
        # Train server model
        self._train_server_model()
        
        # Calibrate for Jetson
        self._calibrate_jetson_model()
        
        logger.info(f"{self.phase} model training complete")
    
    def predict_latency(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """
        Predict latency for given token counts.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with prediction results
        """
        if self.phase == "prefill":
            tokens = input_tokens
        else:  # decode
            tokens = output_tokens
        
        self._validate_inputs(tokens, tokens)  # Use same value for validation
        
        if not self._jetson_model:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Prepare features
        features = self._prepare_features(tokens, tokens)
        
        # Make prediction
        result = self._jetson_model.predict(features)
        
        return self._format_prediction_output(result, input_tokens, output_tokens)
    
    def _setup_data_loading(self) -> None:
        """Setup data loading components."""
        self._data_setup = create_default_data_setup(
            self.config.data_config.server_data_path,
            self.config.data_config.jetson_data_path
        )
    
    def _load_and_preprocess_data(self) -> None:
        """Load and preprocess training data."""
        logger.info(f"Loading data for {self.config.model_config.name}")
        
        # Load raw data
        loader = self._data_setup["data_loader"]
        self._server_data = loader.load_server_data(self.config.model_config.name)
        self._jetson_data = loader.load_jetson_data(self.config.model_config.name)
        
        logger.info(f"Loaded {len(self._server_data)} server samples, "
                   f"{len(self._jetson_data)} Jetson samples")
    
    def _train_server_model(self) -> None:
        """Train server-side model."""
        logger.info(f"Training server {self.phase} model")
        
        # Preprocess server data
        server_features_df = self.preprocessor.preprocess_features(self._server_data)
        
        # Get target variable based on phase
        target_type = f"{self.phase}_latency"
        server_targets = self.preprocessor.extract_target_variable(
            self._server_data, target_type
        )
        
        # Create feature matrix
        X_server, feature_names = self.preprocessor.create_feature_matrix(server_features_df)
        
        # Create and train model
        models = get_default_latency_models()
        model_type = self.config.regression_config.type.value.lower()
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not available")
        
        self._server_model = models[model_type]
        self._server_model.set_feature_names(feature_names)
        self._server_model.fit(X_server, server_targets)
        
        # Evaluate server model
        server_metrics = self.evaluator.evaluate_model(
            server_targets, 
            self._server_model.predict(X_server).predictions
        )
        logger.info(f"Server {self.phase} model metrics: {server_metrics}")
    
    def _calibrate_jetson_model(self) -> None:
        """Calibrate server model for Jetson hardware."""
        logger.info(f"Calibrating {self.phase} model for Jetson")
        
        # Preprocess Jetson data
        jetson_features_df = self.preprocessor.preprocess_features(self._jetson_data)
        
        # Get target variable
        target_type = f"{self.phase}_latency"
        jetson_targets = self.preprocessor.extract_target_variable(
            self._jetson_data, target_type
        )
        
        # Create feature matrix
        X_jetson, _ = self.preprocessor.create_feature_matrix(jetson_features_df)
        
        # Prepare calibration data (tokens, latency pairs)
        if self.phase == "prefill":
            # For prefill, we need input tokens (estimated from prefill time)
            tokens = jetson_features_df["estimated_input_tokens"].values
        else:
            # For decode, use output tokens
            tokens = jetson_features_df["output_tokens"].values
        
        calibration_data = np.column_stack([tokens, jetson_targets])
        
        # Create calibrator
        self._calibrator = create_calibrator(
            calibration_method=self.config.calibration_config.method.value,
            preprocessor=self.preprocessor # Pass the preprocessor
        )
        self._jetson_model = self._calibrator.calibrate_model(
            self._server_model,
            self._jetson_data,
            self.preprocessor
        )
        
        # Evaluate calibrated model
        jetson_predictions = self._jetson_model.predict(X_jetson).predictions
        jetson_metrics = self.evaluator.evaluate_model(jetson_targets, jetson_predictions)
        logger.info(f"Calibrated {self.phase} model metrics: {jetson_metrics}")
    
    def _validate_inputs(self, input_tokens: int, output_tokens: int) -> None:
        """Validate input parameters."""
        if self.phase == "prefill":
            tokens = input_tokens
        else:
            tokens = output_tokens
        
        if tokens <= 0:
            raise ValueError(f"{self.phase.capitalize()} tokens must be positive, got {tokens}")
        
        token_range = self.config.model_config.expected_token_range
        if self.phase == "prefill":
            min_tokens = token_range["min_input_tokens"]
            max_tokens = token_range["max_input_tokens"]
        else:
            min_tokens = token_range["min_output_tokens"]
            max_tokens = token_range["max_output_tokens"]
        
        if tokens < min_tokens or tokens > max_tokens:
            logger.warning(f"{self.phase.capitalize()} tokens ({tokens}) outside expected range "
                         f"[{min_tokens}, {max_tokens}], prediction may be unreliable")
    
    def _prepare_features(self, input_tokens: int, output_tokens: int) -> np.ndarray:
        """Prepare feature vector for prediction."""
        if self.phase == "prefill":
            features = np.array([[input_tokens]])
        else:
            features = np.array([[output_tokens]])
        
        return features
    
    def _format_prediction_output(self, result: PredictionResult,
                                 input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Format prediction results for output."""
        output = {
            f"{self.phase}_latency_seconds": float(result.predictions[0]),
            f"{self.phase}_tokens": input_tokens if self.phase == "prefill" else output_tokens,
            "model_name": self.config.model_config.name,
            "phase": self.phase
        }
        
        # Add confidence intervals if available
        if result.confidence_intervals:
            lower, upper = result.confidence_intervals
            output[f"{self.phase}_latency_lower_bound"] = float(lower[0])
            output[f"{self.phase}_latency_upper_bound"] = float(upper[0])
        
        return output


class CompletePipelineTrainer:
    """Complete training pipeline for all models and phases."""
    
    def __init__(self, output_dir: Path = Path("outputs")):
        """
        Initialize pipeline trainer.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track trained predictors
        self.predictors: Dict[str, Dict[str, PhaseSpecificLatencyPredictor]] = {}
    
    def train_all_models(self, model_sizes: list = None) -> Dict[str, Any]:
        """
        Train predictors for all model sizes and phases.
        
        Args:
            model_sizes: List of ModelSize enums to train, or None for all
            
        Returns:
            Dictionary with training results
        """
        if model_sizes is None:
            model_sizes = [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]
        
        logger.info(f"Training predictors for {len(model_sizes)} model sizes")
        
        results = {}
        
        for model_size in model_sizes:
            logger.info(f"Training models for {model_size.value}")
            
            # Train both prefill and decode predictors
            model_results = self._train_model_size(model_size)
            results[model_size.value] = model_results
        
        # Save consolidated results
        output_path = save_experiment_results(results, self.output_dir, "complete_training")
        logger.info(f"Training complete - results saved to {output_path}")
        
        return results
    
    def _train_model_size(self, model_size: ModelSize) -> Dict[str, Any]:
        """Train both prefill and decode predictors for a model size."""
        from configs.qwen_1_5b import QWEN_1_5B_LINEAR_EXPERIMENT
        from configs.llama_8b import LLAMA_8B_LINEAR_EXPERIMENT  
        from configs.qwen_14b import QWEN_14B_LINEAR_EXPERIMENT
        
        # Get appropriate config
        config_map = {
            ModelSize.SMALL: QWEN_1_5B_LINEAR_EXPERIMENT,
            ModelSize.MEDIUM: LLAMA_8B_LINEAR_EXPERIMENT,
            ModelSize.LARGE: QWEN_14B_LINEAR_EXPERIMENT
        }
        
        config = config_map[model_size]
        
        # Train prefill and decode predictors
        prefill_predictor = PhaseSpecificLatencyPredictor(config, "prefill")
        decode_predictor = PhaseSpecificLatencyPredictor(config, "decode")
        
        # Execute training
        prefill_predictor.train_models()
        decode_predictor.train_models()
        
        # Store predictors
        model_key = model_size.value
        self.predictors[model_key] = {
            "prefill": prefill_predictor,
            "decode": decode_predictor
        }
        
        return {
            "prefill_trained": True,
            "decode_trained": True,
            "model_size": model_size.value
        }
    
    def get_predictor(self, model_size: str, phase: str) -> PhaseSpecificLatencyPredictor:
        """Get trained predictor for specific model size and phase."""
        if model_size not in self.predictors:
            raise ValueError(f"Model size {model_size} not trained")
        
        if phase not in self.predictors[model_size]:
            raise ValueError(f"Phase {phase} not available for {model_size}")
        
        return self.predictors[model_size][phase]
