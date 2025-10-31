# scripts/make_small_datasets.py
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

train_small = train.sample(n=3000, random_state=42)
test_small = test.sample(n=1000, random_state=42)

train_small.to_csv(data_dir / "train.csv", index=False)
test_small.to_csv(data_dir / "test.csv", index=False)

print("Created small dataset samples:")
print(f"train_small.csv → {len(train_small)} rows")
print(f"test_small.csv  → {len(test_small)} rows")
