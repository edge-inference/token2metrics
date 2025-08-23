# Energy Analysis System


## 🔧 Installation

```bash
pip install pandas openpyxl matplotlib seaborn numpy
```

## 🚀 Usage

### Basic Energy Analysis
```bash
python -m energy.cli --base-dir ../dataset/synthetic/gpu/decode/fine
```

### Energy-Performance Correlation
```bash
python -m energy.cli --correlate --energy-dir ../datasets/synthetic/gpu/decode/fine \
  --performance-file ../datasets/synthetic/gpu/decode/fine/processed_results/all_results_by_model_*.xlsx
```

### Power Insights & Visualizations
```bash
python -m energy.cli --insights --verbose
```

### Fitting Analysis
```bash

```

### Advanced Options
```bash
python -m energy.cli --correlate --energy-dir ../dataset/figure3 --performance-file ./results.xlsx

python -m energy.cli --fitting --plot-individual

python -m energy.cli --insights --verbose
```

## 📈 Metrics

- **Energy (J/token)**: Energy consumption per decode token processed
- **Power (W)**: Average and peak power consumption 
- **Efficiency**: Energy consumption vs model size and decode token count
- **Speed**: Decode tokens processed per second

## 📁 Output Files

- Energy summary: `energy_results/energy_model_summary.xlsx`
- Correlation data: `energy_results/energy_performance_correlation_*.xlsx`
- Fit models: `fitting/fitting_summary.json`
- Visualizations: `energy_results/insight_charts/*.png`, `fitting/*.pdf`

For more details on implementation, refer to the code documentation in each module. 