"""
Plotting utilities for model fit diagnostics.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional

def plot_regression_fit(
    tokens: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    phase: str,
    model_name: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot actual vs. predicted latency for regression fit.
    Args:
        tokens: Token counts (input or output)
        actual: Actual latency values
        predicted: Predicted latency values
        phase: 'prefill' or 'decode'
        model_name: Model name for labeling
        save_path: If provided, save plot to this path
        title: Optional plot title
        show: If True, display the plot interactively
    """
    plt.figure(figsize=(7, 5))
    n = len(tokens)
    plt.scatter(tokens, actual, color="blue", alpha=0.5, label=f"Actual (n={n})")
    plt.plot(tokens, predicted, color="red", lw=2, label="Predicted (fit)")
    plt.xlabel(f"{'Input' if phase == 'prefill' else 'Output'} Tokens")
    plt.ylabel(f"{phase.capitalize()} Latency (s)")
    plt.title(title or f"{model_name} {phase.capitalize()} Regression Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_multiple_model_predictions(
    models_data: list,
    phase: str = "decode",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot predictions from multiple models on the same plot with legends.
    Args:
        models_data: List of dicts with keys: 'tokens', 'actual', 'predicted', 'label', 'color' (optional)
        phase: 'prefill' or 'decode'
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        save_path: If provided, save plot to this path
        show: If True, display the plot interactively
    """
    plt.figure(figsize=(8, 6))
    for model in models_data:
        tokens = model['tokens']
        actual = model.get('actual')
        predicted = model.get('predicted')
        label = model.get('label', 'Model')
        color = model.get('color', None)
        # Plot actual as scatter, predicted as line
        if actual is not None:
            plt.scatter(tokens, actual, alpha=0.4, label=f"{label} Actual", color=color)
        if predicted is not None:
            plt.plot(tokens, predicted, lw=2, label=f"{label} Predicted", color=color)
    plt.xlabel(xlabel or ("Input Tokens" if phase == "prefill" else "Output Tokens"))
    plt.ylabel(ylabel or f"{phase.capitalize()} Latency (s)")
    plt.title(title or f"Model Comparison: {phase.capitalize()} Phase")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
