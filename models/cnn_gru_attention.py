"""CNN-GRU with multi-head temporal attention (single attention head block)."""
from typing import Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class CNNGRUAttention(nn.Module):
 
    def __init__(self, input_dim: int, cnn_channels: int = 32, gru_hidden: int = 64, num_heads: int = 4):
        super().__init__()
        # input_dim: number of features per timestep
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(cnn_channels, gru_hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=gru_hidden, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(gru_hidden, 1)

    def forward(self, x):
        # x: (B, seq_len, features)
        x_t = x.transpose(1, 2)  # (B, features, seq_len)
        c = self.relu(self.conv1(x_t))  # (B, cnn_channels, seq_len)
        c = c.transpose(1, 2)  # (B, seq_len, cnn_channels)
        gru_out, _ = self.gru(c)  # (B, seq_len, gru_hidden)
        attn_out, attn_weights = self.attn(gru_out, gru_out, gru_out)  # (B, seq_len, gru_hidden), weights (B, seq_len, seq_len)
        out = self.fc(attn_out[:, -1, :])  # use last time-step
        return out.squeeze(-1), attn_weights.detach().cpu().numpy()


class CNNGRUAttentionModule(pl.LightningModule):
    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNNGRUAttention(input_dim)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        y, _ = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
