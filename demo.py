"""
Demo script showing how to use the Token2Metrics framework.
"""

import logging
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append('.')

from src.utils.helpers import setup_logging
from src.modeling.pipeline import CompletePipelineTrainer
from src.core.config import ModelSize


def main():
    """Run the demo."""
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Token2Metrics Demo")
    
    # Create output directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = CompletePipelineTrainer(output_dir=output_dir)
        
        # Train models for one size (start small for demo)
        logger.info("Training models for Qwen-1.5B...")
        results = trainer.train_all_models([ModelSize.SMALL])
        
        # Make sample predictions
        logger.info("Making sample predictions...")
        
        # Get trained predictors
        prefill_predictor = trainer.get_predictor("1.5B", "prefill")
        decode_predictor = trainer.get_predictor("1.5B", "decode")
        
        # Sample predictions
        test_cases = [
            (50, 20),   # 50 input tokens, 20 output tokens
            (100, 50),  # 100 input tokens, 50 output tokens
            (200, 100)  # 200 input tokens, 100 output tokens
        ]
        
        logger.info("Sample Predictions:")
        logger.info(f"{'Input':<8} {'Output':<8} {'Prefill':<10} {'Decode':<10} {'Total':<10}")
        logger.info("-" * 50)
        
        for input_tokens, output_tokens in test_cases:
            # Get predictions
            prefill_result = prefill_predictor.predict_latency(input_tokens, output_tokens)
            decode_result = decode_predictor.predict_latency(input_tokens, output_tokens)
            
            prefill_time = prefill_result['prefill_latency_seconds']
            decode_time = decode_result['decode_latency_seconds']
            total_time = prefill_time + decode_time
            
            logger.info(f"{input_tokens:<8} {output_tokens:<8} {prefill_time:<10.4f} "
                       f"{decode_time:<10.4f} {total_time:<10.4f}")
        
        logger.info("Demo completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
