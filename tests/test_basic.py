"""
Test basic imports and functionality.
"""

import sys
sys.path.append('.')

def test_basic_imports():
    """Test that basic imports work."""
    try:
        from src.core.config import ModelSize, HardwareType, RegressionType
        from src.core.interfaces import ModelMetrics, PredictionResult
        from src.utils.helpers import setup_logging
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_creation():
    """Test creating basic configurations."""
    try:
        from src.core.config import ModelConfig, ModelSize
        
        config = ModelConfig(
            name="TestModel",
            size=ModelSize.SMALL,
            parameter_count="1.5B",
            expected_token_range={
                "min_input_tokens": 1,
                "max_input_tokens": 1000,
                "min_output_tokens": 1,
                "max_output_tokens": 500
            }
        )
        
        assert config.name == "TestModel"
        assert config.size == ModelSize.SMALL
        print("✅ Configuration creation successful")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_loading_setup():
    """Test data loading setup."""
    try:
        from src.data.registry import register_data_components
        from src.core.factory import DataLoaderFactory
        
        register_data_components()
        available = DataLoaderFactory.get_available_strategies()
        print(f"✅ Data loading setup successful - available loaders: {len(available)}")
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run basic tests."""
    print("Token2Metrics Basic Tests")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Config Creation", test_config_creation),
        ("Data Loading Setup", test_data_loading_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        if test_func():
            passed += 1
    
    print(f"\n{'='*40}")
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All basic tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
