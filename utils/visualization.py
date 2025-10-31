"""Plot helpers for predictions and attention heatmaps."""
from typing import Optional
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, save_path: str, title: str = "Actual vs Predicted") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted', alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved prediction plot to {save_path}")


def plot_attention_heatmap(attention: np.ndarray, save_path: str, x_labels: Optional[list] = None, y_labels: Optional[list] = None) -> None:
    """
    attention: 2D array (seq_len, seq_len) or aggregated attention
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention, xticklabels=x_labels, yticklabels=y_labels, cmap='viridis')
    plt.title('Attention Heatmap')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved attention heatmap to {save_path}")
