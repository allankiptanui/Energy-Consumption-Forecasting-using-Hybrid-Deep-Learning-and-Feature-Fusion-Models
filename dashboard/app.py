# dashboard/app.py
"""
Simple Streamlit dashboard to visualize results and metrics.

Run it with:
    streamlit run dashboard/app.py

Displays:
- Forecast vs Actual plot
- Attention weights plot
- Summary metrics table
- Logs viewer
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOG_DIR = RESULTS_DIR / "logs"

st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")
st.title("⚡ Energy Forecasting Results Dashboard")

# --- Sidebar controls ---
st.sidebar.header("Navigation")
selected_model = st.sidebar.selectbox(
    "Select model",
    ["cnn_gru_attn", "lstm", "gru", "arima"],
)

# --- Metrics ---
metrics_file = METRICS_DIR / f"metrics_{selected_model}.json"
if metrics_file.exists():
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    st.subheader("📊 Evaluation Metrics")
    st.json(metrics)
else:
    st.warning(f"No metrics found for {selected_model}")

# --- Plots ---
fig_forecast = FIG_DIR / f"forecast_{selected_model}.png"
fig_attn = FIG_DIR / f"attention_{selected_model}.png"

st.subheader("📈 Forecast vs Actual")
if fig_forecast.exists():
    st.image(str(fig_forecast))
else:
    st.info("No forecast plot available yet.")

st.subheader("🧭 Temporal Attention")
if fig_attn.exists():
    st.image(str(fig_attn))
else:
    st.info("No attention plot found for this model.")

# --- Logs viewer ---
st.subheader("📝 Training Log (tail)")
log_file = LOG_DIR / f"{selected_model}_train_log.txt"
if log_file.exists():
    with open(log_file, "r") as f:
        lines = f.readlines()[-20:]
    st.text("".join(lines))
else:
    st.info("No training log available yet.")
