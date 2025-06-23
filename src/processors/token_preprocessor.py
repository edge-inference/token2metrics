"""
Concrete implementation of data preprocessors following the Strategy pattern.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

from ..core.interfaces import DataPreprocessor
from ..data.processing import DataCleaner, DataTransformer, DataValidator

logger = logging.getLogger(__name__)


class TokenBasedPreprocessor(DataPreprocessor):
    """Preprocessor specialized for token-based inference metrics."""
    
    def __init__(self, target_type: str = "decode_latency"):
        """
        Initialize preprocessor.
        
        Args:
            target_type: Type of target variable to extract
        """
        self.target_type = target_type
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.validator = DataValidator()
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data into model-ready features.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame with features
        """
        logger.info(f"Preprocessing features for {len(df)} records")
        
        # Determine data source type
        if "decode_time" in df.columns:
            # Server data
            cleaned_df = self.cleaner.clean_server_data(df)
            features = self.transformer.extract_features_server(cleaned_df)
        else:
            # Jetson data
            cleaned_df = self.cleaner.clean_jetson_data(df)
            features = self.transformer.extract_features_jetson(cleaned_df)
        
        # Convert to DataFrame for consistency
        feature_df = pd.DataFrame(features)
        
        logger.info(f"Generated {len(feature_df.columns)} features for {len(feature_df)} samples")
        return feature_df
    
    def extract_target_variable(self, df: pd.DataFrame, target_type: str) -> np.ndarray:
        """
        Extract target variable from DataFrame.
        
        Args:
            df: Preprocessed DataFrame
            target_type: Type of target to extract
            
        Returns:
            Target variable array
        """
        logger.info(f"Extracting target variable: {target_type}")
        
        # Clean data first
        if "decode_time" in df.columns:
            # Server data
            cleaned_df = self.cleaner.clean_server_data(df)
            targets = self.transformer.extract_targets_server(cleaned_df)
        else:
            # Jetson data
            cleaned_df = self.cleaner.clean_jetson_data(df)
            targets = self.transformer.extract_targets_jetson(cleaned_df)
        
        if target_type not in targets:
            available = list(targets.keys())
            raise ValueError(f"Target {target_type} not available. Options: {available}")
        
        target_array = targets[target_type]
        logger.info(f"Extracted {len(target_array)} target values")
        
        return target_array
    
    def create_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix from preprocessed DataFrame.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info("Creating feature matrix...")
        
        # Select relevant feature columns
        feature_columns = self._select_feature_columns(df)
        
        # Extract feature matrix
        X = df[feature_columns].values
        feature_names = feature_columns
        
        # Validate feature matrix
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Feature matrix contains NaN or Inf values, applying imputation")
            X = self._handle_invalid_values(X)
        
        logger.info(f"Created feature matrix: {X.shape}")
        return X, feature_names
    
    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select appropriate feature columns based on available data."""
        available_cols = df.columns.tolist()
        
        # Priority order for feature selection
        preferred_features = [
            "output_tokens",
            "input_tokens", 
            "estimated_input_tokens",
            "total_tokens",
            "token_ratio",
            "log_output_tokens",
            "log_input_tokens"
        ]
        
        # Select features that are available
        selected_features = [
            col for col in preferred_features 
            if col in available_cols
        ]
        
        # Ensure we have at least output_tokens
        if "output_tokens" not in selected_features:
            raise ValueError("output_tokens feature is required but not available")
        
        logger.info(f"Selected features: {selected_features}")
        return selected_features
    
    def _handle_invalid_values(self, X: np.ndarray) -> np.ndarray:
        """Handle NaN and Inf values in feature matrix."""
        # Replace NaN with median
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            if np.isnan(col).any():
                median_val = np.nanmedian(col)
                X[:, col_idx] = np.where(np.isnan(col), median_val, col)
        
        # Replace Inf with max finite value
        X = np.where(np.isposinf(X), np.finfo(np.float64).max, X)
        X = np.where(np.isneginf(X), np.finfo(np.float64).min, X)
        
        return X


class PrefillDecodePreprocessor(TokenBasedPreprocessor):
    """Specialized preprocessor for separate prefill and decode modeling."""
    
    def __init__(self, phase: str = "decode"):
        """
        Initialize for specific inference phase.
        
        Args:
            phase: "prefill" or "decode"
        """
        if phase not in ["prefill", "decode"]:
            raise ValueError(f"Phase must be 'prefill' or 'decode', got {phase}")
        
        self.phase = phase
        target_map = {
            "prefill": "prefill_latency",
            "decode": "decode_latency"
        }
        super().__init__(target_type=target_map[phase])
    
    def create_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create phase-specific feature matrix."""
        if self.phase == "prefill":
            # For prefill, focus on input token features
            preferred_features = [
                "input_tokens",
                "estimated_input_tokens", 
                "log_input_tokens"
            ]
        else:
            # For decode, focus on output token features
            preferred_features = [
                "output_tokens",
                "log_output_tokens",
                "token_ratio"
            ]
        
        available_cols = df.columns.tolist()
        selected_features = [
            col for col in preferred_features 
            if col in available_cols
        ]
        
        if not selected_features:
            # Fallback to basic token features
            if self.phase == "prefill" and "input_tokens" in available_cols:
                selected_features = ["input_tokens"]
            elif self.phase == "decode" and "output_tokens" in available_cols:
                selected_features = ["output_tokens"]
            else:
                raise ValueError(f"No suitable features for {self.phase} phase")
        
        X = df[selected_features].values
        
        # Handle invalid values
        if np.isnan(X).any() or np.isinf(X).any():
            X = self._handle_invalid_values(X)
        
        logger.info(f"Created {self.phase} feature matrix: {X.shape}")
        return X, selected_features
