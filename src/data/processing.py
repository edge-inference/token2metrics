"""
Data processing utilities for cleaning and transforming datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Utility class for cleaning and preprocessing raw data."""
    
    @staticmethod
    def clean_server_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean server-side data removing outliers and invalid records.
        
        Args:
            df: Raw server DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning server data: {len(df)} initial records")
        
        # Create copy to avoid modifying original
        cleaned = df.copy()
        
        # Remove records with missing critical values
        critical_cols = ["input_tokens", "output_tokens", "decode_time", "total_time_ms"]
        initial_count = len(cleaned)
        cleaned = cleaned.dropna(subset=critical_cols)
        logger.info(f"Removed {initial_count - len(cleaned)} records with missing values")
        
        # Remove invalid token counts
        cleaned = cleaned[
            (cleaned["input_tokens"] > 0) & 
            (cleaned["output_tokens"] > 0)
        ]
        
        # Remove invalid timing values
        cleaned = cleaned[
            (cleaned["decode_time"] > 0) & 
            (cleaned["total_time_ms"] > 0)
        ]
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in ["decode_time", "total_time_ms"]:
            mean_val = cleaned[col].mean()
            std_val = cleaned[col].std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            before_count = len(cleaned)
            cleaned = cleaned[
                (cleaned[col] >= lower_bound) & 
                (cleaned[col] <= upper_bound)
            ]
            removed = before_count - len(cleaned)
            if removed > 0:
                logger.info(f"Removed {removed} outliers from {col}")
        
        logger.info(f"Final cleaned data: {len(cleaned)} records")
        return cleaned
    
    @staticmethod
    def clean_jetson_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Jetson calibration data.
        
        Args:
            df: Raw Jetson DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning Jetson data: {len(df)} initial records")
        
        cleaned = df.copy()
        
        # Remove records with missing values
        critical_cols = ["output_tokens", "prefill", "decode", "inference_time"]
        initial_count = len(cleaned)
        cleaned = cleaned.dropna(subset=critical_cols)
        logger.info(f"Removed {initial_count - len(cleaned)} records with missing values")
        
        # Remove invalid values
        cleaned = cleaned[
            (cleaned["output_tokens"] > 0) & 
            (cleaned["prefill"] > 0) & 
            (cleaned["decode"] > 0) & 
            (cleaned["inference_time"] > 0)
        ]
        
        logger.info(f"Final cleaned Jetson data: {len(cleaned)} records")
        return cleaned


class DataTransformer:
    """Transform data for modeling purposes."""
    
    @staticmethod
    def extract_features_server(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract features from server data.
        
        Args:
            df: Cleaned server DataFrame
            
        Returns:
            Dictionary with feature arrays
        """
        features = {
            "input_tokens": df["input_tokens"].values,
            "output_tokens": df["output_tokens"].values,
            "token_ratio": (df["output_tokens"] / df["input_tokens"]).values,
            "total_tokens": (df["input_tokens"] + df["output_tokens"]).values
        }
        
        # Add derived features
        features["log_input_tokens"] = np.log1p(features["input_tokens"])
        features["log_output_tokens"] = np.log1p(features["output_tokens"])
        
        return features
    
    @staticmethod
    def extract_targets_server(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract target variables from server data.
        
        Args:
            df: Cleaned server DataFrame
            
        Returns:
            Dictionary with target arrays
        """
        targets = {
            "decode_latency": df["decode_time"].values,
            "total_latency": df["total_time_ms"].values / 1000.0,  # Convert to seconds
            # Calculate prefill latency: total time - decode time
            "prefill_latency": (df["total_time_ms"] - df["decode_time"]).values / 1000.0 # Convert to seconds
        }
        
        # Add derived targets
        if "tokens_per_second" in df.columns:
            targets["throughput"] = df["tokens_per_second"].values
        
        return targets
    
    @staticmethod
    def extract_features_jetson(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract features from Jetson data.
        
        Args:
            df: Cleaned Jetson DataFrame
            
        Returns:
            Dictionary with feature arrays
        """
        features = {
            "output_tokens": df["output_tokens"].values
        }
        
        # Add input tokens if available (estimated from prefill timing)
        if "prefill" in df.columns:
            # Estimate input tokens from prefill time (rough approximation)
            features["estimated_input_tokens"] = np.maximum(
                1, df["prefill"].values / 10  # Rough estimate
            )
            features["token_ratio"] = (
                features["output_tokens"] / features["estimated_input_tokens"]
            )
        
        features["log_output_tokens"] = np.log1p(features["output_tokens"])
        
        return features
    
    @staticmethod
    def extract_targets_jetson(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract target variables from Jetson data.
        
        Args:
            df: Cleaned Jetson DataFrame
            
        Returns:
            Dictionary with target arrays
        """
        targets = {
            "decode_latency": df["decode"].values / 1000.0,  # Convert to seconds
            "prefill_latency": df["prefill"].values / 1000.0,  # Convert to seconds
            "total_latency": df["inference_time"].values
        }
        
        if "tokens_per_second" in df.columns:
            targets["throughput"] = df["tokens_per_second"].values
        
        return targets


class DataValidator:
    """Validate processed data quality."""
    
    @staticmethod
    def validate_feature_target_alignment(features: Dict[str, np.ndarray], 
                                        targets: Dict[str, np.ndarray]) -> bool:
        """
        Validate that features and targets have consistent shapes.
        
        Args:
            features: Feature dictionary
            targets: Target dictionary
            
        Returns:
            True if aligned
            
        Raises:
            ValueError: If shapes don't align
        """
        if not features or not targets:
            raise ValueError("Features and targets cannot be empty")
        
        # Get sample lengths
        feature_lengths = [len(arr) for arr in features.values()]
        target_lengths = [len(arr) for arr in targets.values()]
        
        # Check all features have same length
        if len(set(feature_lengths)) > 1:
            raise ValueError(f"Feature arrays have inconsistent lengths: {feature_lengths}")
        
        # Check all targets have same length
        if len(set(target_lengths)) > 1:
            raise ValueError(f"Target arrays have inconsistent lengths: {target_lengths}")
        
        # Check features and targets align
        if feature_lengths[0] != target_lengths[0]:
            raise ValueError(
                f"Feature length ({feature_lengths[0]}) != target length ({target_lengths[0]})"
            )
        
        logger.info(f"Data validation passed: {feature_lengths[0]} samples")
        return True
    
    @staticmethod
    def check_data_quality(features: Dict[str, np.ndarray], 
                          targets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate data quality report.
        
        Args:
            features: Feature dictionary
            targets: Target dictionary
            
        Returns:
            Quality report dictionary
        """
        report = {
            "n_samples": len(next(iter(features.values()))),
            "n_features": len(features),
            "n_targets": len(targets),
            "feature_stats": {},
            "target_stats": {}
        }
        
        # Feature statistics
        for name, arr in features.items():
            report["feature_stats"][name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "has_nan": bool(np.isnan(arr).any()),
                "has_inf": bool(np.isinf(arr).any())
            }
        
        # Target statistics
        for name, arr in targets.items():
            report["target_stats"][name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "has_nan": bool(np.isnan(arr).any()),
                "has_inf": bool(np.isinf(arr).any())
            }
        
        return report
