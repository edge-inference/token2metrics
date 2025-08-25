"""
Common configuration and path management for token2metrics modules.
Reads from files/results.yaml to provide consistent path handling.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Token2MetricsConfig:
    """Centralized configuration for all token2metrics modules."""
    
    def __init__(self):
        self._config = None
        self._repo_root = None
        
    @property
    def repo_root(self) -> Path:
        """Get repository root directory."""
        if self._repo_root is None:
            # Go up from third_party/token2metrics/config/ to repo root
            self._repo_root = Path(__file__).resolve().parents[3]
        return self._repo_root
    
    @property
    def config(self) -> Dict[str, Any]:
        """Load and cache results.yaml configuration."""
        if self._config is None:
            config_path = self.repo_root / "files" / "results.yaml"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        self._config = yaml.safe_load(f)
                except Exception:
                    self._config = {"results": {}}
            else:
                # Fallback empty config
                self._config = {"results": {}}
        return self._config
    
    def get_data_dir(self, category: str, subcategory: str) -> Path:
        """Get input data directory for a category/subcategory."""
        try:
            input_dir = self.config["results"][category][subcategory]["input_dir"]
            return self.repo_root / input_dir
        except KeyError:
            # Fallback to data/{category}/{subcategory}/
            return self.repo_root / "data" / category / subcategory
    
    def get_processed_dir(self, category: str, subcategory: str) -> Path:
        """Get processed output directory for a category/subcategory."""
        try:
            output_dir = self.config["results"][category][subcategory]["output_dir"]
            return self.repo_root / output_dir
        except KeyError:
            # Fallback to data/{category}/{subcategory}/processed/
            return self.repo_root / "data" / category / subcategory / "processed"
    
    def get_outputs_dir(self, module_name: str) -> Path:
        """Get outputs directory for a specific module."""
        return self.repo_root / "outputs" / module_name
    
    def get_model_names(self) -> Dict[str, str]:
        """Get model name mappings."""
        return self.config.get("model_names", {})


# Global config instance
_config = Token2MetricsConfig()


class PathManager:
    """
    Universal PathManager for token2metrics modules.
    Each module can set its module_name to get appropriate paths.
    """
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._output_dir = None
    
    @classmethod
    def set_output_dir(cls, output_dir: str) -> None:
        """Set custom output directory (overrides defaults)."""
        cls._output_dir = Path(output_dir)
    
    def get_output_dir(self) -> Path:
        """Get output directory for this module."""
        if self._output_dir is not None:
            return self._output_dir
        
        # Use centralized outputs/{module_name}/
        output_dir = _config.get_outputs_dir(self.module_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_data_path(self, filename: str) -> Path:
        """Get path for a data file."""
        data_dir = self.get_output_dir() / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """Get path for an output file."""
        output_dir = self.get_output_dir()
        return output_dir / filename
    
    def get_chart_path(self, filename: str, chart_type: str = "general") -> Path:
        """Get path for a chart file."""
        chart_dir = self.get_output_dir() / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        return chart_dir / filename
    
    def ensure_dirs(self) -> None:
        """Create all standard directories."""
        self.get_output_dir().mkdir(parents=True, exist_ok=True)
        (self.get_output_dir() / "data").mkdir(parents=True, exist_ok=True)
        (self.get_output_dir() / "charts").mkdir(parents=True, exist_ok=True)


# Module-specific path managers
def get_prefill_paths():
    """Get paths for prefillenergy module."""
    return {
        'input_dir': _config.get_data_dir('synthetic', 'prefill'),
        'processed_dir': _config.get_processed_dir('synthetic', 'prefill'),
        'output_dir': _config.get_outputs_dir('prefill'),
        'path_manager': PathManager('prefill')
    }


def get_decode_paths():
    """Get paths for decodenergy module."""
    return {
        'input_dir': _config.get_data_dir('synthetic', 'decode'),
        'processed_dir': _config.get_processed_dir('synthetic', 'decode'),
        'output_dir': _config.get_outputs_dir('decode'),
        'path_manager': PathManager('decode')
    }


def get_scaling_paths():
    """Get paths for scalingenergy module."""
    return {
        'input_dir': _config.get_data_dir('synthetic', 'scaling'),
        'processed_dir': _config.get_processed_dir('synthetic', 'scaling'),
        'output_dir': _config.get_outputs_dir('scaling'),
        'path_manager': PathManager('scaling')
    }


def get_decodetokens_paths():
    """Get paths for decodetokens module."""
    return {
        'input_dir': _config.repo_root / "data" / "synthetic" / "gpu" / "decode",
        'output_dir': _config.get_outputs_dir('decodetokens'),
        'path_manager': PathManager('decodetokens')
    }


def get_prefilltokens_paths():
    """Get paths for prefilltokens module."""
    return {
        'input_dir': _config.repo_root / "data" / "synthetic" / "gpu" / "prefill",
        'output_dir': _config.get_outputs_dir('prefilltokens'),
        'path_manager': PathManager('prefilltokens')
    }


# Convenience functions
def get_repo_root() -> Path:
    """Get repository root path."""
    return _config.repo_root


def get_model_names() -> Dict[str, str]:
    """Get model name mappings."""
    return _config.get_model_names()


def get_config() -> Dict[str, Any]:
    """Get full configuration from results.yaml."""
    return _config.config
