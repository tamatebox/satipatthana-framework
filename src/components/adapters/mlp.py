import torch.nn as nn
import torch
from typing import Dict, Any
from src.components.adapters.base import BaseAdapter


class MlpAdapter(BaseAdapter):
    """
    MLP Adapter for Tabular/Flat Data.
    Compresses input (input_dim) -> Latent Space (dim).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get("input_dim")
        if self.input_dim is None:
            raise ValueError("MlpAdapter requires 'input_dim' in config.")

        hidden_dim = config.get("adapter_hidden_dim", 256)

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
