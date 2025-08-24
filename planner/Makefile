# Natural-Plan Benchmark Post Processing Makefile
# Convenient commands for running analysis tasks

.PHONY: help install validate check-missing batch-analysis plots summary clean

# Default target
help:
	@echo "* Natural-Plan Results Processor"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install            - Install Python dependencies"
	@echo "  validate           - Validate results directory structure"
	@echo "  check-missing      - Check for missing evaluations"
	@echo "  score              - Run static-eval across results (batch)"
	@echo "  batch-analysis     - Run comprehensive batch analysis"
	@echo "  full-analysis      - Complete workflow (validate + static eval + analysis)"
	@echo "  plots              - Generate accuracy vs tokens plots only"
	@echo "  summary            - Generate summary report only"
	@echo "  clean              - Clean generated outputs"
	@echo ""
	@echo "Task-specific targets:"
	@echo "  meeting-analysis   - Analyze meeting task only"
	@echo "  calendar-analysis  - Analyze calendar task only"
	@echo "  trip-analysis      - Analyze trip task only"
	@echo ""
	@echo "Usage examples:"
	@echo "  make install"
	@echo "  make validate"
	@echo "  make full-analysis"
	@echo "  make run-static-eval-task TASK=meeting"
	@echo "  make check-missing TASK=meeting"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing processor dependencies..."
	pip install -r requirements.txt

# Validate results structure
validate:
	@echo "- Validating results directory..."
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.processing.validate_results

# Check evaluation coverage
check-coverage:
	@echo "Checking evaluation coverage..."
ifdef TASK
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.processing.check_coverage --results-dir ../results --task $(TASK)
else
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.processing.check_coverage --results-dir ../results
endif

# Run comprehensive batch analysis
batch-analysis: validate
	@echo "- Running comprehensive batch analysis..."
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.main

# Generate plots only
plots:
	@echo "- Generating accuracy vs tokens plots..."
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.analyze_results --plot-type accuracy_vs_tokens

plots-cli:
	@echo "- Generating plots via plots CLI..."
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.plots.cli

# Generate summary only
summary:
	@echo "- Generating summary report..."
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.analyze_results --plot-type summary

# Clean generated outputs
clean:
	@echo " Cleaning generated outputs..."
	rm -rf analysis_output/
	rm -f missing_eval_commands.sh
	@echo "✓ Cleaned!"

# Static evaluation using Google's evaluation scripts
score:
	@echo "- Running static-eval scoring..."
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.processing.score

score-task:
	@echo "- Running static-eval for $(TASK)..."
	PYTHONPATH=../..:$$PYTHONPATH python -m token2metrics.planner.processing.score --task $(TASK)


# Quick analysis for specific task
meeting-analysis:
	@echo "- Analyzing meeting task..."
	python analyze_results.py --task meeting

calendar-analysis:
	@echo "- Analyzing calendar task..."
	python analyze_results.py --task calendar

trip-analysis:
	@echo "- Analyzing trip task..."
	python analyze_results.py --task trip

# Complete workflow including static evaluation
full-analysis: validate run-static-eval batch-analysis
	@echo "✓ Complete analysis workflow finished!"