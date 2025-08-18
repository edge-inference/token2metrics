# Energy Analysis System


## 🔧 Installation

```bash
pip install pandas openpyxl matplotlib seaborn numpy
```

## 🚀 Usage

### Basic Energy Analysis
```bash
# Process raw energy CSV files and generate summary metrics
python -m energy.cli --base-dir ../datasets/synthetic/gpu/prefill_padded/finegrain
```

### Energy-Performance Correlation
```bash
# Correlate energy with performance metrics
python -m energy.cli --correlate --energy-dir ../datasets/synthetic/gpu/prefill_padded/finegrain \
  --performance-file ../datasets/synthetic/gpu/prefill_padded/finegrain/processed_results/all_results_by_model_*.xlsx
```

### Power Insights & Visualizations
```bash
python -m energy.cli --insights --verbose
```

### Fitting Analysis
```bash
# Fit mathematical models to energy and power data
python -m energy.cli --fitting
```

### Advanced Options
```bash
# Specify custom paths
python -m energy.cli --correlate --energy-dir ./custom/path --performance-file ./results.xlsx

# Generate individual model plots
python -m energy.cli --fitting --plot-individual

# Enable verbose output
python -m energy.cli --insights --verbose
```

## 📈 Metrics

- **Energy (J/token)**: Energy consumption per token processed
- **Power (W)**: Average and peak power consumption 
- **Efficiency**: Energy consumption vs model size and token count
- **Speed**: Tokens processed per second

## 📁 Output Files

- Energy summary: `results/energy_model_summary.xlsx`
- Correlation data: `results/energy_performance_correlation_*.xlsx`
- Fit models: `results/fitting/fitting_summary.json`
- Visualizations: `results/insight_charts/*.png`, `results/fitting/*.pdf`

For more details on implementation, refer to the code documentation in each module. 