"""
Demo script showing how to use the Token2Metrics framework.
"""

import logging
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append('.')

from src.utils.helpers import setup_logging
from src.modeling.pipeline import CompletePipelineTrainer
from src.core.config import ModelSize

# Check for logging disable flag (env or CLI)
NO_LOG = os.environ.get("TOKEN2METRICS_NO_LOG", "1") == "1" or "--no-log" in sys.argv

if not NO_LOG:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("token2metrics_demo.log"),
            logging.StreamHandler()
        ]
    )


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
        decode_predictor = trainer.get_predictor("1.5B", "decode")
        
        # Sample predictions
        test_cases = [
            (50, 20),   
            (100, 50), 
            (200, 100) 
        ]
        
        logger.info("Sample Predictions:")
        logger.info(f"{'Input':<8} {'Output':<8} {'Prefill':<10} {'Decode':<10} {'Total':<10}")
        logger.info("-" * 50)
        
        for input_tokens, output_tokens in test_cases:
            # Get predictions
            decode_result = decode_predictor.predict_latency(input_tokens, output_tokens)
            
            decode_time = decode_result['decode_latency_seconds']
            total_time = decode_time

            logger.info(f"{input_tokens:<8} {output_tokens:<8}  "
                       f"{decode_time:<10.4f} {total_time:<10.4f}")
        
        logger.info("Demo completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
