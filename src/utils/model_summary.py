"""
Utility for writing model parameter summaries to a JSON file.
"""
import json
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger("token2metrics")

def write_model_params_summary(
    model_name: str,
    phase: str,
    model_type: str,
    coefs: Any,
    intercept: Any,
    is_jetson: bool,
    scaling_factor: float = None,
    summary_path: str = "outputs/model_params_summary.json"
) -> None:
    """
    Write model parameters to a summary JSON file for diagnostics.
    For Jetson models, also write scaling factor and effective coefficients.
    """
    summary_path = Path(summary_path)
    try:
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
        else:
            summary = {}
    except Exception as e:
        logger.error(f"[model_summary] Failed to read summary file: {e}")
        summary = {}
    key = f"{model_name}_{phase}_{'jetson' if is_jetson else 'server'}_{model_type}"
    debug_msg = f"[model_summary] Writing model params for {key} (coefs={coefs}, intercept={intercept}, scaling_factor={scaling_factor})"
    logger.debug(debug_msg)
    logger.debug(f"[model_summary] Keys before update: {list(summary.keys())}")
    try:
        if is_jetson and scaling_factor is not None:
            # Compute effective Jetson coefficients
            base_coefs = coefs.tolist() if hasattr(coefs, 'tolist') else list(coefs)
            base_intercept = float(intercept)
            jetson_coefs = [scaling_factor * c for c in base_coefs]
            jetson_intercept = scaling_factor * base_intercept
            formula = f"latency_jetson = {scaling_factor:.4g} * (" + \
                      " + ".join([f"{c:.4g}*tokens" for c in base_coefs]) + \
                      f" + {base_intercept:.4g})"
            summary[key] = {
                "phase": phase,
                "hardware": "jetson",
                "scaling_factor": scaling_factor,
                "base_coefficients": base_coefs,
                "base_intercept": base_intercept,
                "jetson_coefficients": jetson_coefs,
                "jetson_intercept": jetson_intercept,
                "formula": formula
            }
        else:
            summary[key] = {
                "phase": phase,
                "hardware": "server",
                "coefficients": coefs.tolist() if hasattr(coefs, 'tolist') else list(coefs),
                "intercept": float(intercept)
            }
    except Exception as e:
        logger.error(f"[model_summary] Could not write model params for {key}: {e}")
        logger.error(f"[model_summary] Problematic coefs: {repr(coefs)} (type: {type(coefs)})")
        logger.error(f"[model_summary] Problematic intercept: {repr(intercept)} (type: {type(intercept)})")
        if is_jetson:
            logger.error(f"[model_summary] Jetson entry for {key} was NOT written to summary file!")
        return
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[model_summary] Model params for {key} written to {summary_path}")
        logger.debug(f"[model_summary] Keys after update: {list(summary.keys())}")
        if is_jetson:
            logger.info(f"[model_summary] Jetson entry for {key} successfully written.")
    except Exception as e:
        logger.error(f"[model_summary] Failed to write summary file: {e}")
        if is_jetson:
            logger.error(f"[model_summary] Jetson entry for {key} was NOT written to summary file!")
