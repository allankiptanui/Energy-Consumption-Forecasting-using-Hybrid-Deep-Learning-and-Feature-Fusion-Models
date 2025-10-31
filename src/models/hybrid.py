 

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseModel


class AttentionBlock(nn.Module):
    """Simple additive attention for sequence weighting."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1, bias=False)
        )

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        weights = F.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(weights * x, dim=1)   # weighted sum
        return context, weights.squeeze(-1)       # (batch, hidden_dim), (batch, seq_len)


class HybridModel(BaseModel):
    """CNN + GRU + Attention Hybrid Model."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 64,
        cnn_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_attention: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_attention = use_attention
        self.static_dim = static_dim

        # CNN feature extractor
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding="same"
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GRU for sequential modeling
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Optional attention layer
        self.attn = AttentionBlock(hidden_dim) if use_attention else None

        # Placeholder for FC layers — defined lazily after we know real input dims
        self.fc1 = None
        self.fc2 = None

        self.to(self.device)

    def _lazy_init_fc(self, input_dim: int):
        """Dynamically initialize FC layers if not already defined."""
        if self.fc1 is None:
            self.fc1 = nn.Linear(input_dim, input_dim // 2 if input_dim > 4 else input_dim)
            self.fc2 = nn.Linear(self.fc1.out_features, 1)
            self.to(self.device)

    def forward(self, seq_x, static_x=None):
        """
        seq_x: (batch, seq_len, input_dim)
        static_x: (batch, static_dim) or None
        """
        # CNN expects (batch, input_dim, seq_len)
        x = seq_x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        # GRU expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        out_seq, _ = self.gru(x)

        # Attention (or last time-step)
        if self.use_attention:
            context, attn_weights = self.attn(out_seq)
        else:
            context = out_seq[:, -1, :]
            attn_weights = None

        # Combine with static features if provided
        if static_x is not None and static_x.ndim == 2 and static_x.shape[1] > 0:
            combined = torch.cat([context, static_x], dim=1)
        else:
            combined = context

        # Lazy initialize FC layers based on combined shape
        self._lazy_init_fc(combined.shape[1])

        y = self.relu(self.fc1(combined))
        y = self.fc2(y).squeeze(-1)

        return (y, attn_weights) if self.use_attention else y


 


