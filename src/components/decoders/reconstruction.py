import torch.nn as nn
import torch
from typing import Dict, Any
from src.components.decoders.base import BaseDecoder


class ReconstructionDecoder(BaseDecoder):
    """
    Simple MLP Decoder for Input Reconstruction.
    Latent Space (dim) -> Output (input_dim).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        output_dim = config.get("input_dim")  # Assuming reconstruction target is same as input
        if output_dim is None:
            raise ValueError("ReconstructionDecoder requires 'input_dim' in config (target dimension).")

        hidden_dim = config.get("decoder_hidden_dim", config.get("adapter_hidden_dim", 64))

        self.net = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # No final activation (linear), assuming standardized data or regression target
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)
