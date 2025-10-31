
 

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, n_features: int, n_static: int = 0, hidden: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super(GRUModel, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.n_static = n_static

        # GRU for sequential data
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Combine GRU output with static features (if any)
        input_to_fc = hidden + n_static if n_static > 0 else hidden

        self.fc = nn.Sequential(
            nn.Linear(input_to_fc, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # one-step forecast
        )

    def forward(self, seq_x, static_x=None):
        """
        seq_x: Tensor of shape (batch, seq_len, n_features)
        static_x: Tensor of shape (batch, n_static) or None
        """
        # GRU output: only keep last hidden state
        out, _ = self.gru(seq_x)
        last_out = out[:, -1, :]  # last time step

        # Concatenate static features if present
        if static_x is not None and self.n_static > 0:
            combined = torch.cat([last_out, static_x], dim=1)
        else:
            combined = last_out

        pred = self.fc(combined)
        return pred

 
