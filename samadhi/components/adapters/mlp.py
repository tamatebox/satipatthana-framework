import torch.nn as nn
import torch
from typing import Dict, Any, Union
from samadhi.components.adapters.base import BaseAdapter
from samadhi.configs.adapters import MlpAdapterConfig  # Changed import path
from samadhi.configs.factory import create_adapter_config  # Added for dict support if needed


class MlpAdapter(BaseAdapter):
    """
    MLP Adapter for Tabular/Flat Data.
    Compresses input (input_dim) -> Latent Space (dim).
    """

    def __init__(self, config: MlpAdapterConfig):
        # Allow legacy dict config
        if isinstance(config, dict):
            config = create_adapter_config(config)

        super().__init__(config)

        self.input_dim = self.config.input_dim
        hidden_dim = self.config.adapter_hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.dim),
            nn.Tanh(),  # Normalize to [-1, 1] for latent space stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
