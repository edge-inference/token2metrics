# Token2Metrics
Translating tokens to latency, energy and cost metrics for edge inference

## Overview
Token2Metrics is a modular framework for predicting LLM inference latency (and, in the future, energy/power) on edge devices (e.g., Jetson) using only token counts. It supports phase-specific modeling (prefill and decode), robust configuration, and extensible regression/calibration strategies.

## Phase-Specific Modeling
- **Prefill latency**: Model predicts prefill time from input tokens only.
- **Decode latency**: Model predicts decode time from output tokens only.
- Each phase is trained and calibrated separately for each model size.

## Quick Start

### Train all models
```bash
python main.py train --all-models --output ./outputs
```

### Train specific model
```bash
python main.py train --model 1.5B --output ./outputs
```

### Make predictions
```bash
python main.py predict --model 1.5B --input-tokens 100 --output-tokens 50
```

### Run demo
```bash
python demo.py
```

## Configuration & Tuning
All modeling, regression, and calibration settings are controlled via config files in `configs/` (e.g., `configs/qwen_1_5b.py`).

- **Regression tuning**: Edit the `RegressionConfig` in the config file to change model type (linear, polynomial, etc.) and hyperparameters (e.g., degree for polynomial, fit_intercept for linear).
- **Calibration tuning**: Edit the `CalibrationConfig` to change the calibration method (e.g., `SIMPLE_SCALING`, `LINEAR_FIT`).
- **Phase selection**: The pipeline automatically models both prefill and decode phases separately.

Example (in `configs/qwen_1_5b.py`):
```python
LINEAR_REGRESSION_CONFIG = RegressionConfig(
    type=RegressionType.LINEAR,
    hyperparameters={
        "fit_intercept": True,
        "normalize": False
    },
    cross_validation_folds=5,
    test_size=0.2,
    random_state=42
)

CALIBRATION_CONFIG = CalibrationConfig(
    method=CalibrationMethod.SIMPLE_SCALING
)
```

To tune, simply edit these values and re-run training or the demo.

## Adding New Experiments
- Copy an existing config file in `configs/`, modify as needed, and import it in your pipeline or CLI.
- You can add new regression types, calibration methods, or model sizes by extending the config and codebase.

## Data Format
See the top of the codebase or the project documentation for expected server and Jetson data columns.

## Support
- For advanced tuning, see the docstrings in `src/core/config.py` and the config files in `configs/`.
- For new regression/calibration strategies, implement a new class and register it in the factory/registry modules.