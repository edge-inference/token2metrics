# Analytical Model Framework for Token-Based Inference Metrics

---

## Scalable, Modular Code Structure and Coding Practices

- **Tokenization helpers**
- **Server‐side latency model fitting**
- **Edge‐device (e.g. Jetson) calibration**
- **Multi‐metric prediction** (latency, energy, power) — _start with latency_

### Design Patterns & Practices

- **Factory** for model instantiation
- **Strategy** for different regression backends
- **Dependency injection** for hardware profilers
- **Data classes** for configuration
- **Clear module separation**
- **Type hints everywhere**
- **Single Responsibility Principle**

---

## Project Objective

Build an analytical modeling framework to predict inference metrics (latency, energy, power) from token counts:

- **Input tokens** (`L_in`) and **output tokens** (`L_out`)
- **Server-side regression model** + **edge calibration** for Jetson

---

## Desired Code Structure

### Suggested Modular Layout

- `modeling/` → regression & calibration classes
- `data/` → modules for loading data 
- `processors/` → preprocessing
- `configs/` → configuration files for different models 1.5B, 7B, 15B, sizes.
- `cli.py` or `main.py` → orchestrator entrypoint
- `tests/` → unit and integration tests

### Design Patterns

- **Factory:** instantiate different regression strategies
- **Strategy:** interchangeable regression backends (linear, polynomial, tree)
- **Dependency Injection:** pass profiler and hardware configs into calibrator
- **Data Classes:** for configuration objects (model & hardware)

### Coding Practices

- **Type hints** for all functions and classes
- **Docstrings** in Google or NumPy style
- **Single Responsibility Principle:** each module/class does one thing
- **No hardcoded magic numbers:** use constants or configs
- **Logging** vs. `print()` for runtime info
- **Never use inline comments**; prefer docstrings and clear yet concise function names

---

## CI & Quality

- **Formatting:** Black
- **Testing:** pytest with fixtures
