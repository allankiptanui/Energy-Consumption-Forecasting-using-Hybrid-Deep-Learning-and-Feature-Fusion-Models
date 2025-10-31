"""Evaluate saved models on test set; compute and save MAE, RMSE, MAPE; save predictions and plots."""
from typing import Dict
import logging
import os
import numpy as np
import joblib
from utils.data_loader import load_train_val_test, read_config
from utils.evaluation import compute_metrics, save_metrics, save_predictions, save_metrics_table
from utils.visualization import plot_predictions, plot_attention_heatmap
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def inv_scale_target(scaler, scaled_vals: np.ndarray) -> np.ndarray:
    """Inverse transform only the first column (target)."""
    if scaler is None:
        return scaled_vals
    n_features = scaler.mean_.shape[0]
    tmp = np.zeros((len(scaled_vals), n_features))
    tmp[:, 0] = scaled_vals
    inv = scaler.inverse_transform(tmp)[:, 0]
    return inv


def evaluate_all(cfg_path: str = "config/config.yaml"):
    cfg = read_config(cfg_path)
    data = load_train_val_test(cfg_path)
    X_test = data['X_test']
    y_test = data['y_test']
    scaler_path = os.path.join(cfg['paths']['scalers_dir'], 'feature_scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    results_dir = cfg['paths']['results_dir']
    preds_dir = cfg['paths']['predictions_dir']
    plots_dir = cfg['paths']['plots_dir']
    models_dir = cfg['paths']['models_dir']
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ARIMA (if exists)
    metrics_collection = {}
    try:
        arima_path = os.path.join(models_dir, 'arima_model.pkl')
        if os.path.exists(arima_path):
            from models.baseline_arima import predict_arima
            arima_model = joblib.load(arima_path)
            preds = predict_arima(arima_model, steps=len(y_test))
            # arima is on original units — y_test is scaled; convert y_test to original
            y_true_orig = inv_scale_target(scaler, y_test) if scaler else y_test
            metrics = compute_metrics(y_true_orig, preds)
            metrics_collection['ARIMA'] = metrics
            save_metrics(metrics, os.path.join(results_dir, 'arima_metrics.csv'))
            save_predictions(np.arange(len(preds)), y_true_orig, preds, os.path.join(preds_dir, 'arima_preds.csv'))
    except Exception:
        logger.exception("ARIMA evaluation failed.")

    # Evaluate Lightning models by finding checkpoints
    prefixes = ['lstm', 'gru', 'cnn_gru_attn', 'hybrid']
    for p in prefixes:
        try:
            candidates = sorted([f for f in os.listdir(models_dir) if f.startswith(p)])
            if not candidates:
                logger.warning(f"No saved files for prefix {p}. Skipping.")
                continue
            model_file = os.path.join(models_dir, candidates[0])
            input_dim = X_test.shape[-1]
            # instantiate module
            if p == 'lstm':
                from models.lstm_model import LSTMModel
                module = LSTMModel(input_size=input_dim)
            elif p == 'gru':
                from models.gru_model import GRUModel
                module = GRUModel(input_size=input_dim)
            elif p == 'cnn_gru_attn':
                from models.cnn_gru_attention import CNNGRUAttentionModule
                module = CNNGRUAttentionModule(input_dim=input_dim)
            elif p == 'hybrid':
                from models.hybrid_cnn_gru_fusion import HybridModule
                module = HybridModule(input_features=input_dim)
            else:
                continue

            # try loading checkpoint (Lightning .ckpt) or state dict (.pt/.pth)
            try:
                if model_file.endswith('.ckpt'):
                    module = module.load_from_checkpoint(model_file)
                elif model_file.endswith('.pt') or model_file.endswith('.pth'):
                    sd = torch.load(model_file, map_location='cpu')
                    module.load_state_dict(sd)
                else:
                    # many ModelCheckpoint files are saved without extension; try torch.load
                    try:
                        sd = torch.load(model_file, map_location='cpu')
                        if isinstance(sd, dict):
                            # maybe contains 'state_dict'
                            if 'state_dict' in sd:
                                module.load_state_dict(sd['state_dict'])
                            else:
                                module.load_state_dict(sd)
                    except Exception:
                        logger.debug(f"Could not load weights from {model_file}; using uninitialized model.")
            except Exception:
                logger.debug("Loading checkpoint failed; proceeding with current weights.")

            module.eval()
            X_tensor = torch.tensor(X_test).float()
            with torch.no_grad():
                out = module(X_tensor)
                if isinstance(out, tuple):
                    preds_scaled = out[0].cpu().numpy()
                    # attention maybe out[1]
                else:
                    # single output
                    preds_scaled = out.cpu().numpy()
            preds = inv_scale_target(scaler, preds_scaled) if scaler else preds_scaled
            y_true = inv_scale_target(scaler, y_test) if scaler else y_test
            metrics = compute_metrics(y_true, preds)
            metrics_collection[p.upper()] = metrics
            save_metrics(metrics, os.path.join(results_dir, f'{p}_metrics.csv'))
            save_predictions(np.arange(len(preds)), y_true, preds, os.path.join(preds_dir, f'{p}_preds.csv'))
            plot_predictions(y_true, preds, os.path.join(plots_dir, f'{p}_pred_plot.png'), title=f"{p} Actual vs Predicted")
            # attention heatmap if present
            try:
                # Attempt to get attention by running a small batch
                small = torch.tensor(X_test[:32]).float()
                with torch.no_grad():
                    out_small = module.model(small) if hasattr(module, 'model') else module(small)
                if isinstance(out_small, tuple) and len(out_small) > 1:
                    attn = out_small[1]
                    # avg across batch
                    attn_np = attn.mean(axis=0) if isinstance(attn, np.ndarray) else attn.detach().cpu().numpy().mean(axis=0)
                    plot_attention_heatmap(attn_np, os.path.join(plots_dir, f'{p}_attn.png'))
            except Exception:
                logger.debug("No attention extracted for model %s", p)
        except Exception:
            logger.exception(f"Evaluation failed for prefix {p}")

    # Save metrics comparison table
    try:
        save_metrics_table(metrics_collection, os.path.join(results_dir, 'metrics_comparison.csv'))
    except Exception:
        logger.exception("Failed to save metrics comparison table.")
