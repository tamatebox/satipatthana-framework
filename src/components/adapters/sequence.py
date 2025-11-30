import torch
import torch.nn as nn
import math
from typing import Dict, Any
from src.components.adapters.base import BaseAdapter


class LstmAdapter(BaseAdapter):
    """
    LSTM Adapter for Sequence tasks.
    Input (Batch, Seq_Len, Input_Dim) -> Latent Vector (Batch, Dim).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config.get("input_dim")
        self.seq_len = config.get("seq_len")
        self.hidden_dim = config.get("adapter_hidden_dim", 128)
        self.num_layers = config.get("lstm_layers", 1)

        if self.input_dim is None or self.seq_len is None:
            raise ValueError("Config must contain 'input_dim' and 'seq_len' for LstmAdapter.")

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, Input_Dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (Num_Layers, Batch, Hidden_Dim) -> Take last layer
        last_hidden = h_n[-1]
        z = self.fc(last_hidden)
        return self.activation(z)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TransformerAdapter(BaseAdapter):
    """
    Transformer Adapter for Sequence tasks.
    Uses Transformer Encoder to compress sequence into latent vector.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config.get("input_dim")
        self.seq_len = config.get("seq_len")
        self.hidden_dim = config.get("adapter_hidden_dim", 128)
        self.num_layers = config.get("transformer_layers", 2)
        self.nhead = config.get("transformer_heads", 4)

        if self.input_dim is None or self.seq_len is None:
            raise ValueError("Config must contain 'input_dim' and 'seq_len' for TransformerAdapter.")

        # Input projection to hidden_dim
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=self.seq_len + 100)

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Flatten and map to latent
        # Aggregation strategy: Average Pooling.
        self.to_latent = nn.Linear(self.hidden_dim, self.dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, Input)
        x = self.input_proj(x)  # (Batch, Seq, Hidden)
        x = self.pos_encoder(x)

        # Transformer Pass
        x = self.transformer_encoder(x)  # (Batch, Seq, Hidden)

        # Pooling (Average over sequence)
        x = x.mean(dim=1)  # (Batch, Hidden)

        z = self.to_latent(x)
        return self.activation(z)
