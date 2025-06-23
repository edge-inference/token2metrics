"""
Command-line interface for the Token2Metrics modeling framework.

This module provides the main entry point for training and using
analytical models to predict inference metrics from token counts.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add current directory to Python path
sys.path.append('.')

from src.utils.helpers import setup_logging, create_output_directories
from src.core.config import ModelSize


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Token2Metrics: Analytical modeling framework for inference metrics prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models for all model sizes
  python main.py train --all-models --output ./outputs
  
  # Train specific model
  python main.py train --model 1.5B --regression linear --output ./outputs
  
  # Make predictions
  python main.py predict --model 8B --input-tokens 100 --output-tokens 50
  
  # Evaluate existing model
  python main.py evaluate --model 14B --model-path ./outputs/qwen_14b_linear/models/
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file (optional)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing configuration files"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train analytical models")
    train_parser.add_argument(
        "--model",
        choices=["1.5B", "8B", "14B"],
        help="Model size to train (required if --all-models not specified)"
    )
    train_parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train models for all model sizes"
    )
    train_parser.add_argument(
        "--regression",
        choices=["linear", "polynomial", "random_forest", "all"],
        default="linear",
        help="Regression method to use"
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output directory for trained models and results"
    )
    train_parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Perform cross-validation during training"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make latency predictions")
    predict_parser.add_argument(
        "--model",
        choices=["1.5B", "8B", "14B"],
        required=True,
        help="Model size to use for prediction"
    )
    predict_parser.add_argument(
        "--input-tokens",
        type=int,
        required=True,
        help="Number of input tokens"
    )
    predict_parser.add_argument(
        "--output-tokens",
        type=int,
        required=True,
        help="Number of output tokens"
    )
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to trained model directory"
    )
    predict_parser.add_argument(
        "--regression",
        choices=["linear", "polynomial", "random_forest"],
        default="linear",
        help="Regression method to use"
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    evaluate_parser.add_argument(
        "--model",
        choices=["1.5B", "8B", "14B"],
        required=True,
        help="Model size to evaluate"
    )
    evaluate_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model directory"
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for evaluation results"
    )
    
    # Batch predict command
    batch_parser = subparsers.add_parser("batch-predict", help="Batch prediction from CSV file")
    batch_parser.add_argument(
        "--model",
        choices=["1.5B", "8B", "14B"],
        required=True,
        help="Model size to use"
    )
    batch_parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="CSV file with input_tokens and output_tokens columns"
    )
    batch_parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output CSV file for predictions"
    )
    batch_parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to trained model directory"
    )
    
    return parser


def main() -> int:
    """Main entry point for the CLI application."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=str(args.log_file) if args.log_file else None
    )
    
    try:
        if args.command == "train":
            return handle_train_command(args, logger)
        elif args.command == "predict":
            return handle_predict_command(args, logger)
        elif args.command == "evaluate":
            return handle_evaluate_command(args, logger)
        elif args.command == "batch-predict":
            return handle_batch_predict_command(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.debug(traceback.format_exc())
        return 1


def handle_train_command(args, logger) -> int:
    """Handle the train command."""
    logger.info("Starting model training...")
    
    if not args.model and not args.all_models:
        logger.error("Must specify either --model or --all-models")
        return 1
    
    try:
        from src.modeling.pipeline import CompletePipelineTrainer
        from src.core.config import ModelSize
        
        # Create trainer
        trainer = CompletePipelineTrainer(output_dir=args.output)
        
        # Determine which models to train
        if args.all_models:
            model_sizes = [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]
            logger.info("Training all model sizes")
        else:
            size_map = {"1.5B": ModelSize.SMALL, "8B": ModelSize.MEDIUM, "14B": ModelSize.LARGE}
            model_sizes = [size_map[args.model]]
            logger.info(f"Training model size: {args.model}")
        
        # Execute training
        results = trainer.train_all_models(model_sizes)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        # Print summary
        for model_size, model_results in results.items():
            logger.info(f"✓ {model_size}: Prefill & Decode models trained")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def handle_predict_command(args, logger) -> int:
    """Handle the predict command."""
    logger.info("Making prediction...")
    
    try:
        from src.modeling.pipeline import CompletePipelineTrainer
        from src.utils.helpers import validate_token_inputs
        
        # Validate inputs
        validate_token_inputs(args.input_tokens, args.output_tokens)
        
        # Load trained models (assume they exist in model_path or default location)
        model_path = args.model_path or Path(f"outputs")
        trainer = CompletePipelineTrainer(output_dir=model_path)
        
        # Get predictors
        prefill_predictor = trainer.get_predictor(args.model, "prefill")
        decode_predictor = trainer.get_predictor(args.model, "decode")
        
        # Make predictions
        prefill_result = prefill_predictor.predict_latency(args.input_tokens, args.output_tokens)
        decode_result = decode_predictor.predict_latency(args.input_tokens, args.output_tokens)
        
        # Display results
        logger.info("Prediction Results:")
        logger.info(f"Model: {args.model}")
        logger.info(f"Input Tokens: {args.input_tokens}")
        logger.info(f"Output Tokens: {args.output_tokens}")
        logger.info(f"Prefill Latency: {prefill_result['prefill_latency_seconds']:.4f} seconds")
        logger.info(f"Decode Latency: {decode_result['decode_latency_seconds']:.4f} seconds")
        
        total_latency = (prefill_result['prefill_latency_seconds'] + 
                        decode_result['decode_latency_seconds'])
        logger.info(f"Total Latency: {total_latency:.4f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


def handle_evaluate_command(args, logger) -> int:
    """Handle the evaluate command."""
    logger.info("Evaluating model...")
    
    # Placeholder for actual evaluation logic
    logger.info("Evaluation command received - implementation pending")
    logger.info(f"Model: {args.model}, Model path: {args.model_path}")
    
    return 0


def handle_batch_predict_command(args, logger) -> int:
    """Handle the batch-predict command."""
    logger.info("Starting batch prediction...")
    
    # Placeholder for actual batch prediction logic
    logger.info("Batch prediction command received - implementation pending")
    logger.info(f"Model: {args.model}")
    logger.info(f"Input file: {args.input_file}, Output file: {args.output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
