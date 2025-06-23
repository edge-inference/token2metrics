"""
Concrete implementation of data loaders for server and Jetson data.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ..core.interfaces import DataLoader

logger = logging.getLogger(__name__)


class MMDataLoader(DataLoader):
    """Data loader for MMLU dataset from server and Jetson sources."""
    
    def __init__(self, server_data_path: Path, jetson_data_path: Path):
        self.server_data_path = server_data_path
        self.jetson_data_path = jetson_data_path
        self._validate_paths()
        self._sheet_mapping = self._build_sheet_mapping()
        self._jetson_file_mapping = self._build_jetson_file_mapping()
        self._canonical_names = {
            v.lower(): v for v in set(self._sheet_mapping.values())
        }
    
    def _validate_paths(self) -> None:
        """Validate that data paths exist."""
        if not self.server_data_path.exists():
            raise FileNotFoundError(f"Server data not found: {self.server_data_path}")
        if not self.jetson_data_path.exists():
            raise FileNotFoundError(f"Jetson data path not found: {self.jetson_data_path}")
    
    def load_server_data(self, model_name: str) -> pd.DataFrame:
        """
        Load server data from Excel file for specific model.
        
        Args:
            model_name: Model name to load (e.g., "Qwen-1.5B")
            
        Returns:
            DataFrame with server measurements
        """
        logger.info(f"Loading server data for model: {model_name}")
        key = model_name.lower()
        sheet_name = self._sheet_mapping.get(key)
        if not sheet_name:
            available = list(self._canonical_names.values())
            raise ValueError(f"Model {model_name} not found. Available: {available}")
        try:
            df = pd.read_excel(self.server_data_path, sheet_name=sheet_name)
            logger.info(f"Loaded {len(df)} server records for {model_name}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load server data for {model_name}: {e}")
    
    def load_jetson_data(self, model_name: str) -> pd.DataFrame:
        """
        Load Jetson calibration data for specific model.
        
        Args:
            model_name: Model name to load
            
        Returns:
            DataFrame with Jetson measurements
        """
        logger.info(f"Loading Jetson data for model: {model_name}")
        key = model_name.lower()
        csv_file = self._jetson_file_mapping.get(key)
        if not csv_file:
            available = list(self._canonical_names.values())
            raise ValueError(f"Jetson data for {model_name} not found. Available: {available}")
        filepath = self.jetson_data_path / csv_file
        if not filepath.exists():
            raise FileNotFoundError(f"Jetson file not found: {filepath}")
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
            logger.info(f"Loaded {len(df)} Jetson records for {model_name}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load Jetson data for {model_name}: {e}")
    
    def validate_data_schema(self, df: pd.DataFrame, data_type: str) -> bool:
        """
        Validate DataFrame schema for server or Jetson data.
        
        Args:
            df: DataFrame to validate
            data_type: "server" or "jetson"
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If schema is invalid
        """
        if data_type == "server":
            required_cols = [
                "input_tokens", "output_tokens", "decode_time", "total_time_ms"
            ]
        elif data_type == "jetson":
            required_cols = [
                "output_tokens", "prefill", "decode", "inference_time"
            ]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {data_type} data: {missing_cols}")
        
        return True
    
    def _build_sheet_mapping(self) -> Dict[str, str]:
        """Get mapping from model names to Excel sheet names (all keys lowercased)."""
        mapping = {
            # Canonical names
            "deepseek-r1-distill-qwen-1_5b": "DeepSeek-R1-Distill-Qwen-1_5B",
            "deepseek-r1-distill-llama-8b": "DeepSeek-R1-Distill-Llama-8B",
            "deepseek-r1-distill-qwen-14b": "DeepSeek-R1-Distill-Qwen-14B",
            "l1-qwen-1_5b-max": "L1-Qwen-1_5B-Max",
            # Aliases
            "qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1_5B",
            "qwen_1_5b": "DeepSeek-R1-Distill-Qwen-1_5B", 
            "qwen-1.5": "DeepSeek-R1-Distill-Qwen-1_5B",
            "llama-8b": "DeepSeek-R1-Distill-Llama-8B",
            "llama_8b": "DeepSeek-R1-Distill-Llama-8B",
            "llama-8": "DeepSeek-R1-Distill-Llama-8B",
            "qwen-14b": "DeepSeek-R1-Distill-Qwen-14B",
            "qwen_14b": "DeepSeek-R1-Distill-Qwen-14B",
            "qwen-14": "DeepSeek-R1-Distill-Qwen-14B",
            "l1-qwen-1.5b": "L1-Qwen-1_5B-Max",
            "l1_qwen_1_5b": "L1-Qwen-1_5B-Max"
        }
        return {k.lower(): v for k, v in mapping.items()}
    
    def _build_jetson_file_mapping(self) -> Dict[str, str]:
        """Get mapping from model names to Jetson CSV filenames (all keys lowercased)."""
        mapping = {
            # DeepSeek models - map to existing CSV files
            "deepseek-r1-distill-qwen-1_5b": "profiling_combined_DSR1-Qwen-1.5B.csv",
            "deepseek-r1-distill-qwen-1.5b": "profiling_combined_DSR1-Qwen-1.5B.csv",
            "qwen-1.5b": "profiling_combined_DSR1-Qwen-1.5B.csv",
            "qwen_1_5b": "profiling_combined_DSR1-Qwen-1.5B.csv",
            "qwen-1.5": "profiling_combined_DSR1-Qwen-1.5B.csv",
            "deepseek-r1-distill-llama-8b": "profiling_combined_DSR1-LLama-8B.csv",
            "deepseek-r1-distill-llama-8": "profiling_combined_DSR1-LLama-8B.csv",
            "llama-8b": "profiling_combined_DSR1-LLama-8B.csv",
            "llama_8b": "profiling_combined_DSR1-LLama-8B.csv", 
            "llama-8": "profiling_combined_DSR1-LLama-8B.csv",
            "deepseek-r1-distill-qwen-14b": "profiling_combined_DSR1-Qwen-14B.csv",
            "deepseek-r1-distill-qwen-14": "profiling_combined_DSR1-Qwen-14B.csv",
            "qwen-14b": "profiling_combined_DSR1-Qwen-14B.csv",
            "qwen_14b": "profiling_combined_DSR1-Qwen-14B.csv",
            "qwen-14": "profiling_combined_DSR1-Qwen-14B.csv",
            # L1 model
            "l1-qwen-1.5b": "profiling_combined_L1Max.csv",
            "l1_qwen_1_5b": "profiling_combined_L1Max.csv"
        }
        return {k.lower(): v for k, v in mapping.items()}


class DataLoaderManager:
    """Manager for creating and configuring data loaders."""
    
    def __init__(self, server_path: Path, jetson_path: Path):
        self.server_path = server_path
        self.jetson_path = jetson_path
    
    def create_loader(self, loader_type: str = "mmlu") -> DataLoader:
        """
        Create a data loader instance.
        
        Args:
            loader_type: Type of loader to create
            
        Returns:
            Configured data loader
        """
        if loader_type == "mmlu":
            return MMDataLoader(self.server_path, self.jetson_path)
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")
    
    def list_available_models(self, loader_type: str = "mmlu") -> List[str]:
        """Get list of available canonical model names for a loader type."""
        loader = self.create_loader(loader_type)
        if isinstance(loader, MMDataLoader):
            return list(loader._canonical_names.values())
        return []
