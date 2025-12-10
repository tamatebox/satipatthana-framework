import torch
import torch.nn as nn
import math
from typing import Dict, Any, Union
from satipatthana.components.adapters.base import BaseAdapter
from satipatthana.configs.adapters import LstmAdapterConfig, TransformerAdapterConfig
from satipatthana.configs.factory import create_adapter_config


class LstmAdapter(BaseAdapter):
    """
    LSTM Adapter for Sequence tasks.
    Input (Batch, Seq_Len, Input_Dim) -> Latent Vector (Batch, Dim).
    """

    def __init__(self, config: LstmAdapterConfig):
        if isinstance(config, dict):
            # For LstmAdapter specific initialization from dict, we can rely on create_adapter_config
            # But create_adapter_config might return MlpAdapterConfig if type is missing.
            # So we should ensure type is set or use LstmAdapterConfig.from_dict directly if we are sure.
            # However, utilizing the factory is safer for fallback logic if any.
            # Here, let's assume if dict is passed to LstmAdapter, it's meant for it.
            if "type" not in config:
                config["type"] = "lstm"
            config = create_adapter_config(config)

        super().__init__(config)

        self.input_dim = self.config.input_dim
        self.seq_len = self.config.seq_len
        self.hidden_dim = self.config.adapter_hidden_dim
        self.num_layers = self.config.lstm_layers

        # Config objects usually have defaults, but check if needed logic implies requirements
        # BaseConfig logic handles basic typing.

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

    def __init__(self, config: TransformerAdapterConfig):
        if isinstance(config, dict):
            if "type" not in config:
                config["type"] = "transformer"
            config = create_adapter_config(config)

        super().__init__(config)

        self.input_dim = self.config.input_dim
        self.seq_len = self.config.seq_len
        self.hidden_dim = self.config.adapter_hidden_dim
        self.num_layers = self.config.transformer_layers
        self.nhead = self.config.transformer_heads

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
