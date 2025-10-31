 
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.static_dim = int(static_dim)
        self.lstm_hidden = int(lstm_hidden)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = bool(use_attention)

        # LSTM encoder: expects input (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Optional simple temporal attention (averaging + learned weights)
        if self.use_attention:
            attn_in = self.lstm_hidden * self.num_directions
            self.attn_layer = nn.Sequential(
                nn.Linear(attn_in, attn_in),
                nn.Tanh(),
                nn.Linear(attn_in, 1, bias=False),
            )
        else:
            self.attn_layer = None

        # We'll build the FC head dynamically on the first forward call
        self.fc = None
        self.to(self.device)

    def _build_fc(self, combined_dim: int):
        """Create a small FC head based on the combined feature size."""
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(self.device)

    def _apply_attention(self, seq_out: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        seq_out: (batch, seq_len, hidden*directions)
        returns: context (batch, hidden*directions), weights (batch, seq_len)
        """
        scores = self.attn_layer(seq_out)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(weights * seq_out, dim=1)  # (B, hidden)
        return context, weights.squeeze(-1)

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor = None):
        """
        seq_x: (batch, seq_len, input_dim)
        static_x: (batch, static_dim) or None
        returns: (preds, attn_weights) if attention used, else preds
        """
        # run LSTM
        seq_out, (h_n, c_n) = self.lstm(seq_x)  # seq_out: (B, T, H * directions)

        if self.attn_layer is not None:
            context, attn_weights = self._apply_attention(seq_out)
        else:
            # last timestep from forward direction(s)
            # If bidirectional, combine last forward and last backward:
            if self.bidirectional:
                # h_n shape: (num_layers * num_dirs, B, H)
                # take last layer's forward and backward
                last_layer = -1
                forward_h = h_n[last_layer - 1]  # (B, H)
                backward_h = h_n[last_layer]     # (B, H)
                context = torch.cat([forward_h, backward_h], dim=1)  # (B, 2H)
            else:
                # h_n[-1] shape: (B, H)
                context = h_n[-1]
            attn_weights = None

        # combine with static features if provided
        if static_x is not None and static_x.ndim == 2 and static_x.shape[1] > 0:
            combined = torch.cat([context, static_x], dim=1)
        else:
            combined = context

        # build fc on first forward if needed
        if self.fc is None:
            self._build_fc(combined.size(1))

        out = self.fc(self.dropout_layer(combined) if hasattr(self, "dropout_layer") else combined)
        # ensure output shape is (batch,)
        out = out.view(-1)

        return (out, attn_weights) if attn_weights is not None else out


# small convenience: add dropout attribute used above (keeps student code uniform)
setattr(LSTMModel, "dropout_layer", nn.Dropout(0.2))

 
 