"""
Loads the UCI 'Individual household electric power consumption' dataset.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import os


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase and underscores."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_ucirepo_via_package(uciml_id: int = 235) -> pd.DataFrame:
    """Try fetching dataset via ucimlrepo (UCI Machine Learning Repository)."""
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as e:
        raise ImportError("ucimlrepo not installed. Run: pip install ucimlrepo") from e

    repo = fetch_ucirepo(id=uciml_id)
    df = getattr(repo.data, "features", None)
    if df is None or not isinstance(df, pd.DataFrame):
        if hasattr(repo.data, "to_pandas"):
            df = repo.data.to_pandas()
    if df is None:
        raise ValueError("Could not extract DataFrame from ucimlrepo dataset")

    df = _standardize_column_names(df)
    print(" Loaded dataset via ucimlrepo.")
    return df


def load_local_raw(raw_dir: str = "data/raw", filename: str = "household_power_consumption.txt") -> pd.DataFrame:
    """Load dataset from local file if available."""
    p = Path(raw_dir) / filename
    if not p.exists():
        raise FileNotFoundError(f" Local file not found at {p}.")
    df = pd.read_csv(p, sep=";", low_memory=False, na_values=["?", "NA"])
    df = _standardize_column_names(df)
    print(f"✅ Loaded dataset from {p}")
    return df


def load_raw(use_ucimlrepo: bool = True, raw_dir: str = "data/raw", filename: str = "household_power_consumption.txt", save_copy: bool = True) -> pd.DataFrame:
    """Unified loader: ucimlrepo first, then local fallback."""
    df = None
    if use_ucimlrepo:
        try:
            df = load_ucirepo_via_package()
        except Exception as e:
            print(" ucimlrepo failed:", e)
            print("Falling back to local file.")
            df = load_local_raw(raw_dir, filename)
    else:
        df = load_local_raw(raw_dir, filename)

    if save_copy:
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(raw_dir) / "household_power_consumption.csv"
        df.to_csv(save_path, index=False)
        print(f" Saved a raw copy to {save_path}")
    return df

 
