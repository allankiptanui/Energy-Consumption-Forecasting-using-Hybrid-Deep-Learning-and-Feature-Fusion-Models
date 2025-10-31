import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_output):
        # gru_output: [batch, seq_len, hidden_dim]
        attn_weights = F.softmax(self.attn(gru_output), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * gru_output, dim=1)   # [batch, hidden_dim]
        return context, attn_weights.squeeze(-1)


class CNN_GRU_Attn(nn.Module):
    def __init__(self, input_dim, cnn_channels=32, hidden_size=64, num_layers=1, attn_dim=32):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features per timestep.
        cnn_channels : int
            Number of channels for the CNN feature extractor.
        hidden_size : int
            Hidden dimension of GRU.
        num_layers : int
            Number of GRU layers.
        attn_dim : int
            Internal dimension for attention mechanism.
        """
        super().__init__()

        # --- CNN encoder ---
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # --- GRU ---
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # --- Attention layer ---
        self.attn = AttentionLayer(hidden_size)

        # --- Output layer ---
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, seq, static=None):
        """
        Parameters
        ----------
        seq : torch.Tensor
            [batch, seq_len, input_dim]
        static : torch.Tensor or None
            Ignored (for interface compatibility)
        """
        # CNN expects [batch, channels, seq_len]
        x = seq.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)  # back to [batch, seq_len, channels]

        gru_out, _ = self.gru(cnn_out)  # [batch, seq_len, hidden_size]

        context, attn_weights = self.attn(gru_out)  # [batch, hidden_size], [batch, seq_len]

        output = self.fc(context).squeeze(-1)  # [batch]

        return output, attn_weights
