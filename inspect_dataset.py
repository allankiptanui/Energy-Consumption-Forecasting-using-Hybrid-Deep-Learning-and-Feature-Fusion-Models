"""
inspect_dataset.py

Utility script to load the UCI energy dataset and print key details
before we design preprocessing steps.

It uses the same loader from src/data/loader.py so you can reuse the logic later.
"""

import pandas as pd
from src.data.loader import load_raw, save_dataframe

def inspect_dataset():
    # Load data (from ucimlrepo or local copy)
    df = load_raw(use_ucimlrepo=True)
    print("\n✅ Dataset Loaded Successfully!")
    print("=" * 60)

    # Basic shape info
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values:")
    print(df.isna().sum())

       # Quick descriptive stats (ignore non-numeric columns safely)
    print("\nSummary Statistics (numeric only):")
    print(df.describe(include=[float, int]))

    # Save a snapshot for review
    save_dataframe(df.head(1000), name="dataset_sample_info")

    print("\n✅ First 5 Rows:")
    print(df.head())
    
        # Save a quick textual summary
    with open("data/processed/dataset_summary.txt", "w") as f:
        f.write("Dataset Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Columns:\n")
        f.write(str(df.columns.tolist()) + "\n\n")
        f.write("Data Types:\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("Missing Values:\n")
        f.write(str(df.isna().sum()) + "\n\n")
        f.write("Numeric Summary:\n")
        f.write(str(df.describe(include=[float, int])) + "\n")
    print("\n✅ Summary saved to data/processed/dataset_summary.txt")


    # Optionally return for notebook use
    return df


if __name__ == "__main__":
    inspect_dataset()
