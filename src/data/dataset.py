 
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

class EnergyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_features: List[str], static_features: List[str],
                 target_col: str = "energy_consumption", window: int = 168, horizon: int = 1):
 
        self.df = df.reset_index(drop=True)
        self.seq_features = seq_features
        self.static_features = static_features
        self.target_col = target_col
        self.window = int(window)
        self.horizon = int(horizon)

        self.X_seq, self.X_static, self.y = self._build_windows()

    def _build_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.df[self.seq_features].values
        stat = self.df[self.static_features].values if len(self.static_features) > 0 else np.zeros((len(self.df), 0))
        targets = self.df[self.target_col].values
        Xs, Xs_static, Ys = [], [], []
        n = len(self.df)
        for end in range(self.window, n - self.horizon + 1):
            start = end - self.window
            Xs.append(data[start:end])                  # (window, n_seq_features)
            idx = end + self.horizon - 1
            Xs_static.append(stat[idx])
            Ys.append(targets[idx])
        if len(Xs) == 0:
            # return empty arrays instead of failing
            return np.zeros((0, self.window, len(self.seq_features)), dtype=np.float32), \
                   np.zeros((0, stat.shape[1]), dtype=np.float32), \
                   np.zeros((0,), dtype=np.float32)
        Xs = np.asarray(Xs, dtype=np.float32)
        Xs_static = np.asarray(Xs_static, dtype=np.float32)
        Ys = np.asarray(Ys, dtype=np.float32)
        return Xs, Xs_static, Ys

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.X_seq[idx])         # (window, n_seq_features)
        static = torch.from_numpy(self.X_static[idx])   # (n_static,)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return seq, static, target

 
