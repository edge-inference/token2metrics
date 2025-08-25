# Decode Energy Analysis

## Commands

### Basic Energy Analysis
```bash
python -m energy.cli --base-dir /path/to/data/synthetic/gpu/decode
```

### Energy-Performance Correlation
```bash
python -m energy.cli --correlate --energy-dir /path/to/data/synthetic/gpu/decode \
  --performance-file /path/to/processed/all_results_by_model_*.xlsx
```

### Power Insights
```bash
python -m energy.cli --insights --verbose
```

### Fitting Analysis
```bash
python -m energy.cli --fitting --correlation-file /path/to/correlation.xlsx
```

### Complete Pipeline
```bash
# Run the full pipeline using the run.sh script
./run.sh
```

### Individual Exponential Models
```bash
# Extract exponential power models for each configuration
python empirical_data.py --verbose
```

### Generate Lookup Tables
```bash
# Create lookup tables from correlation data
python generate_lookup_table.py
```

## Output Locations

- Energy results: `outputs/decode/`
- Correlation data: `outputs/decode/energy_performance_correlation.xlsx`
- Fitting results: `outputs/decode/fitting/`
- Charts: `outputs/decode/charts/`
- Empirical data: `outputs/decode/empirical_data.json`
- Lookup tables: `outputs/decode/fitting/decode_*_lookup.json` 