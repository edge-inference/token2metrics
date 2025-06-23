# Token2Metrics
Translating tokens to latency, energy and cost metrics for edge inference

# Train all models
python main.py train --all-models --output ./outputs

# Train specific model
python main.py train --model 1.5B --output ./outputs

# Make predictions
python main.py predict --model 1.5B --input-tokens 100 --output-tokens 50

# Run demo
python demo.py