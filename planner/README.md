# Natural-Plan Benchmark Post Processing

Post-processing toolkit for Natural-Plan evaluation results. Generates accuracy vs tokens plots and analysis reports.

## Quick Start

```bash
# Run complete analysis (recommended)
python -m token2metrics.planner.main

# Check evaluation coverage
python -m token2metrics.planner.processing.coverage

# Run scoring with Google's scripts
python -m token2metrics.planner.processing.score
```

## Key Modules

- `main.py` - Complete analysis workflow
- `processing/coverage.py` - Check evaluation coverage
- `processing/score.py` - Run Google's evaluation scripts
- `analysis.py` - Generate summary reports
- `plots/` - Plotting functionality

## Outputs

```
outputs/planner/
├── plots/
├── evaluation_summary.csv
└── analysis_insights.md
```

## Directory Structure

```
data/planner/
├── base/{model}/{task}_{timestamp}/results_*.json
└── budget/{model}/{tokens}/{task}_{timestamp}/results_*.json
```