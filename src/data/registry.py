"""
Registration of concrete data loading and preprocessing implementations.
"""

from pathlib import Path

from ..core.factory import DataLoaderFactory, PreprocessorFactory
from ..data.loaders import MMDataLoader, DataLoaderManager
from ..processors.token_preprocessor import TokenBasedPreprocessor, PrefillDecodePreprocessor


def register_data_components() -> None:
    """Register all data loading and preprocessing components with factories."""
    
    # Register data loaders
    DataLoaderFactory.register_loader("mmlu", MMDataLoader)
    
    # Register preprocessors
    PreprocessorFactory.register_preprocessor("token_based", TokenBasedPreprocessor)
    PreprocessorFactory.register_preprocessor("prefill_decode", PrefillDecodePreprocessor)


def create_default_data_setup(server_path: Path, jetson_path: Path) -> dict:
    """
    Create default data loading setup.
    
    Args:
        server_path: Path to server data
        jetson_path: Path to Jetson data
        
    Returns:
        Dictionary with configured components
    """
    register_data_components()
    
    # Create data loader manager
    loader_manager = DataLoaderManager(server_path, jetson_path)
    
    # Create default loader
    data_loader = loader_manager.create_loader("mmlu")
    
    # Create default preprocessors
    token_preprocessor = PreprocessorFactory.create_preprocessor("token_based")
    prefill_preprocessor = PreprocessorFactory.create_preprocessor(
        "prefill_decode", phase="prefill"
    )
    decode_preprocessor = PreprocessorFactory.create_preprocessor(
        "prefill_decode", phase="decode"
    )
    
    return {
        "loader_manager": loader_manager,
        "data_loader": data_loader,
        "token_preprocessor": token_preprocessor,
        "prefill_preprocessor": prefill_preprocessor,
        "decode_preprocessor": decode_preprocessor,
        "available_models": loader_manager.list_available_models()
    }


# Convenience function for quick setup
def quick_data_setup() -> dict:
    """Quick setup with default paths."""
    server_path = Path("datasets/server/full_mmlu_by_model.xlsx")
    jetson_path = Path("datasets/tegra")
    return create_default_data_setup(server_path, jetson_path)
