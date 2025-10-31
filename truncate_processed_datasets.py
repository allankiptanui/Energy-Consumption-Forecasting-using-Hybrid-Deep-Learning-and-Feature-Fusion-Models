# truncate_processed_datasets.py
"""
Randomly sample 3000 train entries and 1000 test entries
from the processed datasets and save them as smaller versions
for quick experimentation.
"""

import pandas as pd
from pathlib import Path

# Paths
processed_dir = Path("data/processed")
train_path = processed_dir / "train.csv"
test_path = processed_dir / "test.csv"

# Load the full datasets
print("Loading datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Sample sizes
n_train = 3000
n_test = 1000

# Randomly sample (without replacement)
print(f"Sampling {n_train} rows from train and {n_test} from test...")
train_sample = train_df.sample(n=min(n_train, len(train_df)), random_state=42)
test_sample = test_df.sample(n=min(n_test, len(test_df)), random_state=42)

# Save to new files
train_out = processed_dir / "train_small.csv"
test_out = processed_dir / "test_small.csv"

train_sample.to_csv(train_out, index=False)
test_sample.to_csv(test_out, index=False)

print(f"✅ Done! Saved:")
print(f" - {train_out} ({len(train_sample)} rows)")
print(f" - {test_out} ({len(test_sample)} rows)")
