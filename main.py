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
from src.utils.plotting import plot_multiple_model_predictions
import numpy as np
from src.utils.helpers import setup_logging, create_output_directories, save_experiment_results
from src.core.config import ModelSize
from src.modeling.pipeline import CompletePipelineTrainer
from src.data.registry import create_default_data_setup
from src.modeling.evaluator import LatencyModelEvaluator
from sklearn.model_selection import train_test_split
import os

# Add current directory to Python path
sys.path.append('.')


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
        choices=["1.5B", "8B", "14B", "L1MAX"],
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
    train_parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Manual scaling factor to override Jetson calibration (optional)"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make latency predictions")
    predict_parser.add_argument(
        "--model",
        choices=["1.5B", "8B", "14B", "L1MAX"],
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
    predict_parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Manual scaling factor to override Jetson calibration (optional)"
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    evaluate_parser.add_argument(
        "--model",
        choices=["1.5B", "8B", "14B", "L1MAX"],
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
    evaluate_parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Manual scaling factor to override Jetson calibration (optional)"
    )
    
    # Batch predict command
    batch_parser = subparsers.add_parser("batch-predict", help="Batch prediction from CSV file")
    batch_parser.add_argument(
        "--model",
        choices=["1.5B", "8B", "14B", "L1MAX"],
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
    batch_parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Manual scaling factor to override Jetson calibration (optional)"
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
        # Create trainer
        trainer = CompletePipelineTrainer(output_dir=args.output)
        
        # Determine which models to train
        if args.all_models:
            model_sizes = [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE, "L1MAX"]
            logger.info("Training all model sizes (including L1MAX)")
        else:
            size_map = {"1.5B": ModelSize.SMALL, "8B": ModelSize.MEDIUM, "14B": ModelSize.LARGE, "L1MAX": "L1MAX"}
            model_sizes = [size_map[args.model]]
            logger.info(f"Training model size: {args.model}")
        
        # Execute training
        results = trainer.train_all_models(model_sizes)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        # Print summary
        for model_size, model_results in results.items():
            if isinstance(model_size, str):
                model_key = model_size
            else:
                model_key = model_size.value
            logger.info(f"✓ {model_key}: Prefill & Decode models trained")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def handle_predict_command(args, logger) -> int:
    """Handle the predict command."""
    logger.info("Making prediction...")
    
    try:
        from src.utils.helpers import validate_token_inputs

        # Validate inputs
        validate_token_inputs(args.input_tokens, args.output_tokens)

        # Load trained models (assume they exist in model_path or default location)
        model_path = args.model_path or Path(f"outputs")
        trainer = CompletePipelineTrainer(output_dir=model_path)

        # Only decode predictor (prefill is disabled)
        decode_predictor = trainer.get_predictor(args.model, "decode")
        decode_result = decode_predictor.predict_latency(args.input_tokens, args.output_tokens)

        # Display results
        logger.info("Prediction Results:")
        logger.info(f"Model: {args.model}")
        logger.info(f"Input Tokens: {args.input_tokens}")
        logger.info(f"Output Tokens: {args.output_tokens}")
        logger.info(f"Decode Latency: {decode_result['decode_latency_seconds']:.4f} seconds")

        # Optionally, plot predictions for a range of output tokens
        tokens_range = np.arange(1, 1 + 1000, 10)  # Example: 1 to 1000 tokens
        predictions = []
        for t in tokens_range:
            pred = decode_predictor.predict_latency(args.input_tokens, t)
            predictions.append(pred['decode_latency_seconds'])
        models_data = [{
            'tokens': tokens_range,
            'actual': None,  # No ground truth in prediction mode
            'predicted': predictions,
            'label': f"{args.model} decode"
        }]
        plot_multiple_model_predictions(
            models_data,
            phase="decode",
            title=f"Decode Latency Prediction: {args.model}",
            save_path=f"outputs/prediction_plot_{args.model}_decode.png",
            show=False
        )
        logger.info(f"Prediction plot saved to outputs/prediction_plot_{args.model}_decode.png")
        return 0
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


def handle_evaluate_command(args, logger) -> int:
    """Handle the evaluate command."""
    logger.info("Evaluating model...")
    try:
        # Load trained decode predictor
        model_path = args.model_path or Path(f"outputs")
        trainer = CompletePipelineTrainer(output_dir=model_path)
        decode_predictor = trainer.get_predictor(args.model, "decode")

        # Load Jetson data for the model
        data_setup = create_default_data_setup(
            Path("datasets/server/full_mmlu_by_model.xlsx"),
            Path("datasets/tegra")
        )
        loader = data_setup["data_loader"]
        model_name = decode_predictor.config.model_config.name
        jetson_df = loader.load_jetson_data(model_name)
        preprocessor = decode_predictor.preprocessor
        features_df = preprocessor.preprocess_features(jetson_df)
        X, _ = preprocessor.create_feature_matrix(features_df)
        y_true = preprocessor.extract_target_variable(jetson_df, "decode_latency")

        # Use held-out test split
        test_size = getattr(decode_predictor.config.regression_config, "test_size", 0.2)
        X_train, X_test, y_train, y_test, tokens_train, tokens_test = train_test_split(
            X, y_true, features_df["output_tokens"].values, test_size=test_size, random_state=42
        )

        # Predict with Jetson-calibrated model
        y_pred = decode_predictor._jetson_model.predict(X_test).predictions

        # Evaluate
        evaluator = LatencyModelEvaluator()
        metrics = evaluator.evaluate_model(y_test, y_pred)
        logger.info(f"Jetson evaluation metrics: {metrics}")

        # Plot actual vs. predicted for Jetson test set
        n_test = len(y_test)
        models_data = [{
            'tokens': tokens_test,
            'actual': y_test,
            'predicted': y_pred,
            'label': f"Jetson Predicted (n={n_test})"
        }]
        os.makedirs(args.output, exist_ok=True)
        plot_path = os.path.join(args.output, f"evaluation_plot_{args.model}_decode_jetson.png")
        plot_multiple_model_predictions(
            models_data,
            phase="decode",
            title=f"Jetson Decode Latency Evaluation: {args.model}",
            save_path=plot_path,
            show=False
        )
        logger.info(f"Jetson evaluation plot saved to {plot_path}")

        # Save metrics
        metrics_path = os.path.join(args.output, f"evaluation_metrics_{args.model}_decode_jetson.json")
        save_experiment_results(metrics.__dict__, Path(args.output), f"evaluation_metrics_{args.model}_decode_jetson")
        logger.info(f"Jetson evaluation metrics saved to {metrics_path}")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


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
