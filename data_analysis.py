
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Directories
data_dir = Path("data/processed")
results_dir = Path("results/analysis")
results_dir.mkdir(parents=True, exist_ok=True)

# Load data
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")


print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Combine for global stats
df = pd.concat([train_df, test_df], axis=0)

# Basic statistics
summary = df.describe().T
summary.to_csv(results_dir / "summary_statistics.csv")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(results_dir / "correlation_heatmap.png")
plt.close()

# Target distribution
if "energy_consumption" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df["energy_consumption"], kde=True)
    plt.title("Energy Consumption Distribution")
    plt.xlabel("Energy Consumption (kWh)")
    plt.tight_layout()
    plt.savefig(results_dir / "energy_distribution.png")
    plt.close()
# Time series trend 
time_col = "datetime" if "datetime" in df.columns else None
if time_col:
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_col], df["energy_consumption"], linewidth=0.8)
    plt.title("Energy Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption (kWh)")
    plt.tight_layout()
    plt.savefig(results_dir / "energy_time_trend.png")
    plt.close()

# Pairplot for selected features
selected_features = ["global_active_power", "voltage", "global_intensity", "energy_consumption"]
for f in selected_features:
    if f not in df.columns:
        selected_features.remove(f)

if len(selected_features) > 2:
    sns.pairplot(df[selected_features].sample(1000, random_state=42))
    plt.savefig(results_dir / "pairplot_features.png")
    plt.close()

