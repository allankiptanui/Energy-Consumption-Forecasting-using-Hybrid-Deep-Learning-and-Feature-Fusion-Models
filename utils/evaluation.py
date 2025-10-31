"""Evaluation metrics and saving utilities."""
from typing import Dict
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def save_metrics_table(metrics_dict: Dict[str, Dict[str, float]], out_path: str) -> None:
    """
    metrics_dict: {model_name: {'MAE':.., 'RMSE':.., 'MAPE':..}, ...}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df = df.reset_index().rename(columns={'index': 'model'})
    df.to_csv(out_path, index=False)
    logger.info(f"Saved metrics comparison to {out_path}")


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
    logger.info(f"Saved metrics to {path}")


def save_predictions(index, actual, predicted, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({'index': index, 'actual': actual, 'predicted': predicted})
    df.to_csv(path, index=False)
    logger.info(f"Saved predictions to {path}")
