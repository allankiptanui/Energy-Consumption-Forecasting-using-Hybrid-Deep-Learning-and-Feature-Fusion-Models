"""
Enhanced unified training workflow (v3)
---------------------------------------
Supports both deep learning (PyTorch) and classical (ARIMA) models.
Fixes: ARIMAModel no parameters issue by separating training paths.
"""

import argparse
import json
from pathlib import Path
import torch
import pandas as pd
from src.data.loader import load_raw
from src.data.preprocessor import run_full_preprocessing
from src.training.trainer import train_model
from src.training.evaluator import (
    evaluate_model_on_loader,
    compute_metrics,
    save_metrics_csv,
    plot_forecast,
    plot_loss,
    plot_residuals,
    plot_error_histogram,
    plot_pred_vs_actual_scatter,
    plot_error_qq,
    plot_error_over_time,
    plot_forecast_multi,
    plot_attention
)
from src.utils.logger import get_logger


# ---------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------
def build_model(model_name: str, input_dim: int):
    model_name = model_name.lower()
    if model_name == "gru":
        from src.models.gru import GRUModel as M
    elif model_name == "lstm":
        from src.models.lstm import LSTMModel as M
    elif model_name == "cnn_gru_attn":
        from src.models.cnn_gru_attn import CNN_GRU_Attn as M
    elif model_name == "hybrid":
        from src.models.hybrid import HybridModel as M
    elif model_name == "arima":
        from src.models.arima import ARIMAModel as M
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Try flexible constructors
    for kw in ["input_dim", "input_size", "in_features"]:
        try:
            return M(**{kw: input_dim})
        except TypeError:
            continue
    try:
        return M(input_dim)
    except Exception:
        return M()


# ---------------------------------------------------------------
# Main training + evaluation workflow
# ---------------------------------------------------------------
def main(args):
    logger = get_logger(f"run_{args.model}")
    logger.info(f"Starting training workflow for model={args.model}")

    # ------------------------------------------------------------------ #
    # Load and preprocess
    # ------------------------------------------------------------------ #
    raw_df = load_raw(use_ucimlrepo=False, raw_dir="data/raw", filename="household_power_consumption.txt")
    logger.info(f"Raw dataset shape: {raw_df.shape}")

    train_df, test_df, scaler = run_full_preprocessing(raw_df, processed_dir="data/processed")

    if args.quick:
        train_df = train_df.sample(2000, random_state=42)
        test_df = test_df.sample(800, random_state=42)
        logger.info("⚡ Using QUICK mode (sampled smaller dataset).")

    logger.info(f"Preprocessed train={train_df.shape}, test={test_df.shape}")

    # ------------------------------------------------------------------ #
    # Infer features
    # ------------------------------------------------------------------ #
    all_columns = list(train_df.columns)
    target_col = "energy_consumption"

    seq_features = [
        "global_active_power", "global_reactive_power", "voltage", "global_intensity",
        "sub_metering_1", "sub_metering_2", "sub_metering_3", "hour", "dayofweek", "month"
    ]
    seq_features = [f for f in seq_features if f in all_columns]

    static_features = ["month", "is_weekend", "month_sin", "month_cos", "is_holiday"]
    static_features = [f for f in static_features if f in all_columns]

    from src.data.dataset import EnergyDataset
    train_ds = EnergyDataset(train_df, seq_features, static_features)
    test_ds = EnergyDataset(test_df, seq_features, static_features)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    input_dim = len(seq_features)
    logger.info(f"Detected seq_features={len(seq_features)}, static_features={len(static_features)}")
    logger.info(f"Input dim for model: {input_dim}")

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model = build_model(args.model, input_dim)
    model_name = args.model.lower()
    logger.info(f"Built model: {model.__class__.__name__}")

    # ------------------------------------------------------------------ #
    # Detect ARIMA vs Deep Model
    # ------------------------------------------------------------------ #
    is_classical = hasattr(model, "fit") and hasattr(model, "forecast") and not hasattr(model, "parameters")

    if is_classical:
        # -------------------------------------------------------------- #
        # Classical ARIMA training path
        # -------------------------------------------------------------- #
        logger.info("🧮 Detected classical model (fit/forecast). Running ARIMA-style training...")

        # Fit ARIMA on the target series
        train_y = train_df[target_col].values
        test_y = test_df[target_col].values

        model.fit(train_y)
        preds = model.forecast(steps=len(test_y))

        # Compute metrics
        metrics = compute_metrics(test_y, preds)
        logger.info(f"📊 Metrics for {model_name}: {metrics}")

        # Save metrics and predictions
        Path("results").mkdir(exist_ok=True)
        with open(f"results/metrics_{model_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        save_metrics_csv(metrics, "results/metrics_summary.csv")

        pd.DataFrame({"y_true": test_y, "y_pred": preds}).to_csv(f"results/preds_{model_name}.csv", index=False)

        # Simple forecast plot
        plot_forecast(test_y, preds, model_name=model_name, show=False)
        plot_error_histogram(test_y, preds, model_name=model_name, show=False)
        plot_residuals(test_y, preds, model_name=model_name, show=False)

        logger.info(f"✅ Classical model run complete. Results saved under results/")
        return

    # ------------------------------------------------------------------ #
    # Deep model training path (PyTorch)
    # ------------------------------------------------------------------ #
    cfg = {
        "batch_size": 64,
        "epochs": 3 if args.quick else 10,
        "lr": 1e-3,
        "early_stop_patience": 5,
        "num_workers": 0,
    }

    history, best_model_path = train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=test_ds,
        cfg=cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path=f"results/checkpoints/{args.model}_best.pth",
    )
    logger.info("✅ Training complete")

    # ------------------------------------------------------------------ #
    # Evaluate
    # ------------------------------------------------------------------ #
    y_true, y_pred, attn, true_batches, pred_batches = evaluate_model_on_loader(
        model, test_loader, return_per_batch=True
    )

    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"📊 Metrics for {model_name}: {metrics}")

    # Save metrics + plots
    metrics_path = f"results/metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    save_metrics_csv(metrics, "results/metrics_summary.csv")

    fig_dir = Path(f"results/figures/{model_name}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_forecast(y_true, y_pred, model_name=model_name, show=False)
    plot_loss(history, model_name=model_name, show=False)
    plot_residuals(y_true, y_pred, model_name=model_name, show=False)
    plot_error_histogram(y_true, y_pred, model_name=model_name, show=False)
    plot_pred_vs_actual_scatter(y_true, y_pred, model_name=model_name, show=False)
    plot_error_qq(y_true, y_pred, model_name=model_name, show=False)
    plot_error_over_time(y_true, y_pred, model_name=model_name, show=False)
    if attn is not None:
        plot_attention(attn, model_name=model_name, show=False)
    if true_batches is not None:
        plot_forecast_multi(true_batches, pred_batches, model_name=model_name, show=False)

    logger.info(f"✅ All plots saved under results/figures/{model_name}/")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training + evaluation workflow")
    parser.add_argument("--model", required=True, type=str, help="Model name: gru | lstm | cnn_gru_attn | hybrid | arima")
    parser.add_argument("--quick", action="store_true", help="Use small dataset for debugging")
    args = parser.parse_args()
    main(args)
