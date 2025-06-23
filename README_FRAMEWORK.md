# Token2Metrics: Analytical Modeling Framework

An analytical modeling framework for predicting inference metrics (latency, energy, power) from token counts on edge devices, particularly Jetson systems.

## Project Structure

```
token2metrics/
├── src/                    # Source code
│   ├── core/              # Core configuration and interfaces
│   │   ├── config.py      # Data classes for configuration
│   │   ├── interfaces.py  # Abstract base classes and interfaces  
│   │   └── factory.py     # Factory pattern implementations
│   ├── data/              # Data loading modules
│   ├── processors/        # Data preprocessing modules
│   ├── modeling/          # Regression and calibration classes
│   └── utils/             # Utility functions and helpers
├── configs/               # Configuration files for different models
│   ├── qwen_1_5b.py      # Configuration for 1.5B model
│   ├── llama_8b.py       # Configuration for 8B model  
│   └── qwen_14b.py       # Configuration for 14B model
├── tests/                 # Unit and integration tests
├── datasets/              # Data directories
│   ├── server/           # Server-side measurements
│   └── tegra/            # Jetson/Tegra measurements
├── main.py               # CLI entry point
└── requirements.txt      # Python dependencies
```

## Design Patterns

- **Factory Pattern**: For instantiating different regression strategies
- **Strategy Pattern**: Interchangeable regression backends (linear, polynomial, random forest)
- **Template Method**: Overall modeling workflow structure
- **Dependency Injection**: Hardware profilers and configurations
- **Data Classes**: Immutable configuration objects

## Supported Models

- **Qwen-1.5B**: Small model configuration
- **LLaMA-8B**: Medium model configuration  
- **Qwen-14B**: Large model configuration

## Supported Regression Methods

- Linear Regression
- Polynomial Regression
- Random Forest
- XGBoost (planned)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### CLI Usage

```bash
# Train models for all sizes
python main.py train --all-models --output ./outputs

# Train specific model
python main.py train --model 1.5B --regression linear --output ./outputs

# Make predictions
python main.py predict --model 8B --input-tokens 100 --output-tokens 50

# Evaluate model
python main.py evaluate --model 14B --model-path ./outputs/qwen_14b_linear/models/
```

## Data Format

### Server Data
Expected columns in Excel file:
- `subject`, `question_id`, `question`, `choices`, `correct_answer`
- `predicted_choice`, `is_correct`, `generated_text`
- `ttft`, `decode_time`, `total_time_ms`, `tokens_per_second`
- `input_tokens`, `output_tokens`, `generated_text_length`

### Jetson Data  
Expected columns in CSV files:
- `subset`, `ground_truth_index`, `predicted_choice_letter`
- `full_output`, `inference_time`, `output_tokens`
- `prefill`, `decode`, `correct`, `tokens_per_second`

## Development Status

🚧 **Current Status**: Foundation and architecture complete

### ✅ Completed
- Project structure and modular design
- Configuration system with data classes
- Abstract interfaces and factory patterns
- CLI framework
- Model configurations for 1.5B, 8B, 14B models
- Logging and utility functions
- Basic test structure

### 🔄 Next Steps  
- Implement concrete data loader classes
- Implement preprocessing strategies
- Implement regression strategies (linear, polynomial, RF)
- Implement calibration methods
- Add model evaluation and metrics
- Complete CLI command implementations
- Add visualization utilities

## Architecture Principles

- **Single Responsibility**: Each module/class has one purpose
- **Type Hints**: All functions and classes are fully typed
- **No Magic Numbers**: Constants and configurations used throughout
- **Logging over Print**: Structured logging for runtime information
- **Modular Design**: Clear separation between data, processing, modeling
- **Testable**: Comprehensive test coverage with pytest

## Contributing

This project follows professional coding standards:
- Black formatting
- Type hints everywhere  
- Google-style docstrings
- Comprehensive testing
- Clear module separation
