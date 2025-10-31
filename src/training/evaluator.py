from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

 

def _make_model_plot_path(base_dir: str, model_name: str, plot_name: str) -> str:
 
    base = Path(base_dir) / model_name
    base.mkdir(parents=True, exist_ok=True)
    fname = f"{plot_name}_{model_name}.png"
    return str(base / fname)


def _show_or_save(fig: plt.Figure, save_path: Optional[str], show: bool) -> None:
    """
    Save figure to save_path   and show (if show=True).
    If both, it will save then show.
    """
    if save_path:
        save_path = str(save_path)
        outdir = os.path.dirname(save_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"💾 Saved plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


 
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (in percent), numerically stable."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (in percent)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.clip(denom, 1e-8, None)
    return float(100.0 * np.mean(2 * np.abs(y_pred - y_true) / denom))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return a dictionary with main metrics."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }


def evaluate_model_on_loader(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: Optional[str] = None,
    return_per_batch: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.to(device)
    model.eval()

    true_batches: List[np.ndarray] = []
    pred_batches: List[np.ndarray] = []
    attn_batches: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            # batch expected to be (seq, static, y) or similar
            if len(batch) == 3:
                seq, static, target = batch
            elif len(batch) == 2:
                seq, target = batch
                static = None
            else:
                raise ValueError("DataLoader must yield (seq, static, target) or (seq, target)")

            seq = seq.to(device)
            if static is not None:
                static = static.to(device)
            target = target.to(device)

            # model may return preds or (preds, attn)
            out = model(seq, static) if static is not None else model(seq)
            if isinstance(out, (tuple, list)):
                preds = out[0]
                attn = out[1] if len(out) > 1 else None
            else:
                preds = out
                attn = None

            true_batches.append(target.cpu().numpy())
            pred_batches.append(preds.cpu().numpy())
            if attn is not None:
                try:
                    attn_batches.append(attn.cpu().numpy())
                except Exception:
                    attn_batches.append(np.asarray(attn))

    if not true_batches:
        raise ValueError("DataLoader yielded no batches.")

    y_true = np.concatenate(true_batches, axis=0)
    y_pred = np.concatenate(pred_batches, axis=0)
    attn_arr = np.concatenate(attn_batches, axis=0) if attn_batches else None

    if return_per_batch:
        return y_true, y_pred, attn_arr, true_batches, pred_batches
    return y_true, y_pred, attn_arr, None, None



def plot_loss(
    history: Dict[str, List[float]],
    model_name: Optional[str] = None,
    title: str = "Training / Validation Loss",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot training & validation loss curves."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.get("train_loss", []), label="train")
    ax.plot(history.get("val_loss", []), label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.legend()
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "loss")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    n_plot: int = 300,
    start_idx: int = 0,
    title: str = "Forecast vs Actual",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot actual vs predicted time series overlayed for n_plot points."""
    n = min(len(y_true) - start_idx, len(y_pred) - start_idx, n_plot)
    if n <= 0:
        raise ValueError("Invalid n_plot / start_idx")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    idx = slice(start_idx, start_idx + n)
    ax.plot(y_true[idx], label="Actual", linewidth=1.5, marker=".", markersize=4)
    ax.plot(y_pred[idx], label="Predicted", linewidth=1.2, marker=".", markersize=4)
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.legend()
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "forecast")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_forecast_multi(
    true_batches: List[np.ndarray],
    pred_batches: List[np.ndarray],
    model_name: Optional[str] = None,
    n_samples: int = 5,
    horizon: int = 50,
    title: str = "Multiple Forecast Samples",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
 
    if not true_batches or not pred_batches:
        raise ValueError("true_batches and pred_batches must be provided lists of arrays.")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    n_batches = len(true_batches)
    n_samples = min(n_samples, n_batches)
    idxs = np.random.choice(n_batches, size=n_samples, replace=False)

    for i in idxs:
        t = np.asarray(true_batches[i])[:horizon]
        p = np.asarray(pred_batches[i])[:horizon]
        ax.plot(t, color="steelblue", alpha=0.6, linewidth=1.2)
        ax.plot(p, color="orange", alpha=0.6, linewidth=1.2, linestyle="--")

    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.set_xlabel("Time step in horizon")
    ax.set_ylabel("Value")
    ax.legend(["actual", "predicted"], loc="upper right")
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "forecast_multi")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_attention(
    alpha: Optional[np.ndarray],
    model_name: Optional[str] = None,
    sample_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    title: str = "Temporal Attention",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """Plot attention weights as bar plot for a given sample and head."""
    if alpha is None:
        print("No attention weights to plot.")
        return None, None

    arr = np.asarray(alpha).copy()
    # reduce (B, H, T) -> (B, T) by averaging heads if needed
    if arr.ndim == 3:
        if head_idx is not None:
            arr = arr[:, head_idx, :]
        else:
            arr = arr.mean(axis=1)  # average heads
    # now arr can be (B, T) or (T,) or (B,) etc.
    if arr.ndim == 2:
        if sample_idx is not None:
            to_plot = arr[sample_idx]
        else:
            to_plot = arr.mean(axis=0)
    elif arr.ndim == 1:
        to_plot = arr
    else:
        raise ValueError("Unsupported attention shape.")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(len(to_plot)), to_plot, color="skyblue")
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Weight")
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "attention")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    title: str = "Residuals vs True Values",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter of residuals vs actuals to check bias."""
    resid = np.asarray(y_true) - np.asarray(y_pred)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true, resid, alpha=0.6, s=20)
    ax.axhline(0, color="red", linestyle="--")
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Residual")
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "residuals")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    bins: int = 50,
    title: str = "Error Distribution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Histogram of errors (y_true - y_pred)."""
    err = np.asarray(y_true) - np.asarray(y_pred)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(err, bins=bins, kde=True, ax=ax, color="lightcoral")
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "error_hist")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_pred_vs_actual_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    title: str = "Predicted vs Actual",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter plot of predictions vs actuals with identity line."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, color="red", linestyle="--")
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "pred_vs_actual")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_error_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    title: str = "QQ Plot of Residuals",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """QQ-plot of residuals vs normal distribution."""
    err = np.asarray(y_true) - np.asarray(y_pred)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(err, dist="norm", plot=ax)
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "qq")
    _show_or_save(fig, save_path, show)
    return fig, ax


def plot_error_over_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    title: str = "Absolute Error over Time",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot absolute error over time (index)."""
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(abs_err, color="tomato", linewidth=1.2)
    final_title = f"{model_name.upper()} — {title}" if model_name else title
    ax.set_title(final_title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("|Error|")
    ax.grid(True)

    if save_path is None and model_name:
        save_path = _make_model_plot_path("results/figures", model_name, "error_over_time")
    _show_or_save(fig, save_path, show)
    return fig, ax


def save_metrics_csv(metrics: Dict[str, Any], path: str) -> None:
    """
    Append metrics dictionary as a row to CSV at `path`.
    Adds a timestamp column automatically.
    """
    metrics = dict(metrics)  # shallow copy
    metrics["timestamp"] = pd.Timestamp.now().isoformat()
    df = pd.DataFrame([metrics])
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    mode = "a" if os.path.exists(path) else "w"
    header = not os.path.exists(path)
    df.to_csv(path, mode=mode, header=header, index=False)
    print(f"💾 Appended metrics to {path}")


 