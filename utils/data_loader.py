"""energy-forecasting-hybrid/utils/data_loader.py
Load and preprocess UCI Household Power dataset, align/resample to hourly,
fuse with weather (real file or synthesized), create sequences and save processed arrays.
"""
from typing import Dict, Any
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import yaml
import holidays

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_config(cfg_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        logger.debug("PyTorch not installed for seeding.")


def _parse_household_power(path: str) -> pd.DataFrame:
    """
    Parse the UCI Household Power CSV:
    dataset summary provided indicates columns: date, time, global_active_power, ...
    Convert to datetime and numeric, resample to hourly by mean.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Household power file not found at {path}")
    # read with pandas with flexible parsing
    try:
        # The UCI file often uses ';' separator and 'Date'/'Time' columns.
        df = pd.read_csv(path, sep=';', low_memory=False)
    except Exception:
        # fallback to comma
        df = pd.read_csv(path, low_memory=False)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # If 'date' and 'time' exist, combine
    if 'date' in df.columns and 'time' in df.columns:
        # Ensure date/time strings are properly formatted
        df['datetime'] = df['date'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip()
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        df = df.set_index('datetime')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.set_index('datetime')
    else:
        raise ValueError("Could not find date/time columns in household power file.")

    # Columns of interest
    expected = ['global_active_power', 'global_reactive_power', 'voltage', 'global_intensity',
                'sub_metering_1', 'sub_metering_2', 'sub_metering_3']
    for col in expected:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"Expected column {col} not present in household file; filling with NaNs.")
            df[col] = np.nan

    # Resample to hourly mean
    df_hour = df[expected].resample('H').mean()
    # Interpolate limited gaps, then forward/backfill
    df_hour = df_hour.interpolate(limit=4).ffill().bfill()
    return df_hour


def _load_weather(weather_path: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load weather_hourly.csv if exists (expects 'datetime' column and 'temperature' & 'humidity' columns),
    otherwise synthesize simple seasonality-based temperature & humidity aligned to index.
    """
    if os.path.exists(weather_path):
        try:
            wdf = pd.read_csv(weather_path)
            wdf.columns = [c.strip().lower() for c in wdf.columns]
            if 'datetime' in wdf.columns:
                wdf['datetime'] = pd.to_datetime(wdf['datetime'], errors='coerce')
                wdf = wdf.set_index('datetime')
            elif 'date' in wdf.columns and 'time' in wdf.columns:
                wdf['datetime'] = pd.to_datetime(wdf['date'].astype(str) + ' ' + wdf['time'].astype(str), errors='coerce')
                wdf = wdf.set_index('datetime')
            else:
                raise ValueError("Weather file missing datetime/date+time columns.")
            # expected columns: temperature, humidity
            if 'temperature' not in wdf.columns:
                for alt in ['temp', 't', 'air_temperature']:
                    if alt in wdf.columns:
                        wdf = wdf.rename(columns={alt: 'temperature'})
                        break
            if 'humidity' not in wdf.columns:
                for alt in ['hum', 'relative_humidity']:
                    if alt in wdf.columns:
                        wdf = wdf.rename(columns={alt: 'humidity'})
                        break
            # Resample to hourly to align with power index
            wdf = wdf.resample('H').mean().reindex(index).interpolate(limit=4).ffill().bfill()
            if 'temperature' not in wdf.columns:
                logger.warning("Weather file loaded but 'temperature' missing — synthesizing temp.")
                wdf['temperature'] = 10 + 10 * np.sin(2 * np.pi * (wdf.index.dayofyear / 365.0))
            if 'humidity' not in wdf.columns:
                logger.warning("Weather file loaded but 'humidity' missing — synthesizing humidity.")
                wdf['humidity'] = 50 + 20 * np.cos(2 * np.pi * (wdf.index.hour / 24.0))
            return wdf[['temperature', 'humidity']]
        except Exception as e:
            logger.exception("Failed to parse weather file; synthesizing weather.")
    # Synthesize weather
    logger.info("Synthesizing weather (temperature & humidity).")
    t = 10 + 10 * np.sin(2 * np.pi * (index.dayofyear.values / 365.0)) + 5 * np.sin(2 * np.pi * (index.hour.values / 24.0))
    h = 50 + 20 * np.cos(2 * np.pi * (index.hour.values / 24.0))
    wdf = pd.DataFrame({'temperature': t, 'humidity': h}, index=index)
    return wdf


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic time features and holiday/weekend flags."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df['is_weekend'] = (df.index.weekday >= 5).astype(int)
    try:
        cal = holidays.CountryHoliday('US')
        df['is_holiday'] = df.index.normalize().isin(cal.keys()).astype(int)
    except Exception:
        df['is_holiday'] = 0
    return df


def load_and_process_data(cfg_path: str = "config/config.yaml") -> None:
    """Full pipeline: load raw datasets, fuse, scale, create sequences and save arrays."""
    cfg = read_config(cfg_path)
    set_seed(cfg.get('random_seed', 42))
    raw_dir = cfg['data']['raw_dir']
    processed_dir = cfg['data']['processed_dir']
    fused_csv = cfg['data']['fused_csv']
    train_val_test_path = cfg['data']['train_val_test']
    os.makedirs(processed_dir, exist_ok=True)

    household_path = os.path.join(raw_dir, 'household_power_consumption.csv')
    weather_path = os.path.join(raw_dir, 'weather_hourly.csv')

    logger.info("Loading household power data...")
    power_df = _parse_household_power(household_path)
    logger.info("Loading/creating weather data...")
    weather_df = _load_weather(weather_path, power_df.index)

    # Merge and add features
    df = power_df.join(weather_df, how='left')
    df = add_time_features(df)
    # Ensure target column name consistent with earlier references
    if 'global_active_power' not in df.columns:
        raise KeyError("Target 'global_active_power' missing after processing.")
    # Reorder: place target first
    cols = ['global_active_power'] + [c for c in df.columns if c != 'global_active_power']
    df = df[cols].fillna(method='ffill').fillna(method='bfill')
    # Save fused CSV
    df.to_csv(fused_csv)
    logger.info(f"Saved fused dataset to {fused_csv}")

    # Scale features
    scaler = StandardScaler()
    arr = df.values.astype(float)
    arr_scaled = scaler.fit_transform(arr)
    os.makedirs(cfg['paths']['scalers_dir'], exist_ok=True)
    joblib.dump(scaler, os.path.join(cfg['paths']['scalers_dir'], 'feature_scaler.pkl'))
    logger.info("Saved feature scaler.")

    # Create sequences
    seq_len = cfg['training']['seq_len']
    pred_h = cfg['training']['pred_horizon']
    X = []
    y = []
    n_total = arr_scaled.shape[0]
    for i in range(n_total - seq_len - pred_h + 1):
        X.append(arr_scaled[i:i + seq_len])
        y.append(arr_scaled[i + seq_len:i + seq_len + pred_h, 0])  # target is first column
    X = np.array(X)
    y = np.array(y).squeeze(-1)
    logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")

    # Split: 70% train, 15% val, 15% test
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    np.savez_compressed(train_val_test_path,
                        X_train=X[:train_end],
                        y_train=y[:train_end],
                        X_val=X[train_end:val_end],
                        y_val=y[train_end:val_end],
                        X_test=X[val_end:],
                        y_test=y[val_end:])
    logger.info(f"Saved train/val/test to {train_val_test_path}")
