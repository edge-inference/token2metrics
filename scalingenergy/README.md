# Figure 6 Analysis

## Token Analysis (Main)
```bash
python export.py
python analyze.py
```
Output: `plots/decode_latency_vs_ps.pdf`

## Energy Analysis
```bash
python run_energy_analysis.py
```
Output: `results/energy/` plots + Excel data

### New System Metrics
The energy analysis now extracts and visualizes additional system metrics from energy CSV files:
- **RAM Usage**: Total and used memory (MB), calculated usage percentage
- **GPU Usage**: GPU utilization percentage 
- **Temperature**: Junction temperature (°C)

These metrics are automatically included in:
- Correlation analysis and Excel exports
- Additional plots: `ram_usage_vs_ps.pdf`, `gpu_usage_vs_ps.pdf`, `temperature_vs_ps.pdf`
- **Dual-Axis Plot**: `dual_axis_ps_power_gpu.pdf` showing Power & GPU Utilization vs Parallel Scaling Factor
- System metrics sheets in Excel output files

## Advanced Energy CLI
```bash
python energy_cli.py analyze --energy-dir ../tegra/scaling --performance-file results/energy/scaling_summary.xlsx
```

## Standalone Plotting
```bash
python plot_energy_results.py --input results/energy/energy_correlations.xlsx --output plots/
```

## Dual-Axis Plot Generation
```bash
# Generate dual-axis plot from real correlation data
python plot_dual_axis_only.py --input results/energy/energy_correlations.xlsx --output plots/

# Test dual-axis plot feature with sample data
python test_3d_plot.py
```
Generates dual-axis visualization: `dual_axis_ps_power_gpu.pdf` showing Power & GPU Utilization vs Parallel Scaling Factor

## Requirements
pandas, numpy, matplotlib, scipy, openpyxl 