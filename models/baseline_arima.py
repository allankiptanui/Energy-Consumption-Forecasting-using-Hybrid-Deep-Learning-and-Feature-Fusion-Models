"""ARIMA wrapper using pmdarima.auto_arima"""
from typing import Any
import logging
import joblib
import os
import numpy as np
import pmdarima as pm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_arima(ts: np.ndarray, seasonal: bool = True, m: int = 24, save_path: str = "models/saved/arima_model.pkl") -> Any:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        model = pm.auto_arima(ts, seasonal=seasonal, m=m, suppress_warnings=True, stepwise=True)
        joblib.dump(model, save_path)
        logger.info(f"Saved ARIMA model to {save_path}")
        return model
    except Exception:
        logger.exception("ARIMA training failed.")
        raise
