"""
Complete training pipeline for phase-specific latency predictors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import json

from ..core.interfaces import LatencyPredictor, RegressionStrategy, PredictionResult
from ..core.config import ExperimentConfig, ModelSize
from ..data.registry import create_default_data_setup
from ..modeling.registry import get_default_latency_models
from ..modeling.calibration import create_calibrator
from ..modeling.evaluator import LatencyModelEvaluator
from ..processors.token_preprocessor import PrefillDecodePreprocessor
from ..utils.helpers import ExperimentTracker, save_experiment_results
from ..utils.plotting import plot_regression_fit
from ..utils.model_summary import write_model_params_summary
from .persistence import save_predictor, load_predictor

logger = logging.getLogger("token2metrics")


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
        
        # Manual scaling factor for calibration
        self.manual_scaling_factor = getattr(config.calibration_config, 'manual_scaling_factor', None)
    
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
        # Plot regression fit
        try:
            tokens = server_features_df["input_tokens"].values if self.phase == "prefill" else server_features_df["output_tokens"].values
            preds = self._server_model.predict(X_server).predictions
            plot_regression_fit(
                tokens,
                server_targets,
                preds,
                phase=self.phase,
                model_name=self.config.model_config.name,
                save_path=f"outputs/fit_{self.config.model_config.name}_{self.phase}_server.png",
                title=f"{self.config.model_config.name} {self.phase.capitalize()} Server Fit"
            )
        except Exception as e:
            logger.warning(f"Could not plot server regression fit: {e}")
        
        # Write server model parameters to summary file
        try:
            model_type = self.config.regression_config.type.value.lower()
            if hasattr(self._server_model, 'model') and hasattr(self._server_model.model, 'coef_'):
                coefs = self._server_model.model.coef_
                intercept = self._server_model.model.intercept_
                write_model_params_summary(
                    self.config.model_config.name, self.phase, model_type, coefs, intercept, is_jetson=False
                )
            elif hasattr(self._server_model, 'coef_'):
                write_model_params_summary(
                    self.config.model_config.name, self.phase, model_type, self._server_model.coef_, self._server_model.intercept_, is_jetson=False
                )
        except Exception as e:
            logger.warning(f"Could not write server model parameters: {e}")
    
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
            if "input_tokens" in jetson_features_df.columns:
                tokens = jetson_features_df["input_tokens"].values
            elif "estimated_input_tokens" in jetson_features_df.columns:
                tokens = jetson_features_df["estimated_input_tokens"].values
            else:
                raise ValueError("No input token feature available for Jetson prefill calibration (expected 'input_tokens' or 'estimated_input_tokens').")
        else:
            # For decode, use output_tokens
            tokens = jetson_features_df["output_tokens"].values
        
        calibration_data = np.column_stack([tokens, jetson_targets])
        
        # Create calibrator
        self._calibrator = create_calibrator(
            calibration_method=self.config.calibration_config.method.value,
            preprocessor=self.preprocessor,
            manual_scaling_factor=self.manual_scaling_factor
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
        # Plot regression fit
        try:
            tokens = jetson_features_df["input_tokens"].values if self.phase == "prefill" and "input_tokens" in jetson_features_df.columns else (
                jetson_features_df["estimated_input_tokens"].values if self.phase == "prefill" and "estimated_input_tokens" in jetson_features_df.columns else jetson_features_df["output_tokens"].values
            )
            plot_regression_fit(
                tokens,
                jetson_targets,
                jetson_predictions,
                phase=self.phase,
                model_name=self.config.model_config.name,
                save_path=f"outputs/fit_{self.config.model_config.name}_{self.phase}_jetson.png",
                title=f"{self.config.model_config.name} {self.phase.capitalize()} Jetson Fit"
            )
        except Exception as e:
            logger.warning(f"Could not plot Jetson regression fit: {e}")
        
        # Write Jetson model parameters to summary file
        try:
            model_type = self.config.regression_config.type.value.lower()
            jm = self._jetson_model
            # If ScaledRegressionWrapper, extract scaling and base model params
            if hasattr(jm, 'scaling_factor') and hasattr(jm, 'base_model'):
                scaling = getattr(jm, 'scaling_factor', None)
                base = getattr(jm, 'base_model', None)
                coefs = None
                intercept = None
                if base is not None:
                    if hasattr(base, 'model') and hasattr(base.model, 'coef_'):
                        coefs = base.model.coef_
                        intercept = base.model.intercept_
                    elif hasattr(base, 'coef_'):
                        coefs = base.coef_
                        intercept = base.intercept_
                logger.info(f"Jetson model uses scaling factor: {scaling}")
                logger.info(f"Jetson model base coefficients: {coefs}, intercept: {intercept}")
                write_model_params_summary(
                    self.config.model_config.name, self.phase, model_type,
                    coefs,
                    intercept,
                    is_jetson=True,
                    scaling_factor=scaling
                )
            elif hasattr(jm, 'model') and hasattr(jm.model, 'coef_'):
                coefs = jm.model.coef_
                intercept = jm.model.intercept_
                write_model_params_summary(
                    self.config.model_config.name, self.phase, model_type, coefs, intercept, is_jetson=True
                )
            elif hasattr(jm, 'coef_'):
                write_model_params_summary(
                    self.config.model_config.name, self.phase, model_type, jm.coef_, jm.intercept_, is_jetson=True
                )
            else:
                logger.warning("Jetson model has no coefficients or scaling factor!")
        except Exception as e:
            logger.warning(f"Could not write Jetson model parameters: {e}")
            # Debug: log Jetson model type and attributes before writing summary
            try:
                logger.debug(f"Jetson model type: {type(self._jetson_model)}")
                logger.debug(f"Jetson model dir: {dir(self._jetson_model)}")
            except Exception as e:
                logger.warning(f"Could not inspect Jetson model: {e}")
    
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
    
    def _normalize_model_key(self, model_size) -> str:
        """Normalize model size to string key (e.g., '14B')."""
        if hasattr(model_size, "value"):
            return model_size.value
        if isinstance(model_size, str):
            # Accept both '14B' and 'L1MAX' as valid keys
            return model_size
        raise ValueError(f"Unrecognized model_size: {model_size}")

    def train_all_models(self, model_sizes: list = None) -> Dict[str, Any]:
        """
        Train predictors for all model sizes and phases.
        
        Args:
            model_sizes: List of ModelSize enums or strings to train, or None for all
            
        Returns:
            Dictionary with training results
        """
        if model_sizes is None:
            model_sizes = [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]
        logger.info(f"Training predictors for {len(model_sizes)} model sizes")
        results = {}
        for model_size in model_sizes:
            model_key = self._normalize_model_key(model_size)
            logger.info(f"Training models for {model_key}")
            # Train both prefill and decode predictors
            model_results = self._train_model_size(model_size)
            results[model_key] = model_results
        # Save consolidated results
        output_path = save_experiment_results(results, self.output_dir, "complete_training")
        logger.info(f"Training complete - results saved to {output_path}")
        return results
    
    def _train_model_size(self, model_size) -> Dict[str, Any]:
        from configs.qwen_1_5b import QWEN_1_5B_LINEAR_EXPERIMENT
        from configs.llama_8b import LLAMA_8B_LINEAR_EXPERIMENT  
        from configs.l1max_qwen_1_5b import L1MAX_LINEAR_EXPERIMENT
        from configs.qwen_14b import get_qwen_14b_linear_experiment

        if model_size == "L1MAX":
            config = L1MAX_LINEAR_EXPERIMENT
        elif model_size == ModelSize.SMALL:
            config = QWEN_1_5B_LINEAR_EXPERIMENT
        elif model_size == ModelSize.MEDIUM:
            config = LLAMA_8B_LINEAR_EXPERIMENT
        elif model_size == ModelSize.LARGE:
            config = get_qwen_14b_linear_experiment(scale_factor=getattr(self, 'manual_scaling_factor', None))
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        # Only train decode predictor (skip prefill)
        decode_predictor = PhaseSpecificLatencyPredictor(config, "decode")
        decode_predictor.train_models()
        model_key = self._normalize_model_key(model_size)
        self.predictors[model_key] = {
            "decode": decode_predictor
        }
        # Save predictor to disk
        save_predictor(decode_predictor, model_key, "decode", self.output_dir)
        return {
            "decode_trained": True,
            "model_size": model_key
        }
    
    def get_predictor(self, model_size: str, phase: str) -> PhaseSpecificLatencyPredictor:
        """Get trained predictor for specific model size and phase. Loads from disk if not in memory."""
        model_key = self._normalize_model_key(model_size)
        if model_key in self.predictors and phase in self.predictors[model_key]:
            return self.predictors[model_key][phase]
        # Try loading from disk
        try:
            predictor = load_predictor(model_key, phase, self.output_dir)
            if model_key not in self.predictors:
                self.predictors[model_key] = {}
            self.predictors[model_key][phase] = predictor
            return predictor
        except Exception as e:
            raise ValueError(f"Model size {model_key} with phase {phase} not trained or saved: {e}")
