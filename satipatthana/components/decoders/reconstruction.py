import torch.nn as nn
import torch
from typing import Dict, Any
from satipatthana.components.decoders.base import BaseDecoder
from satipatthana.configs.decoders import ReconstructionDecoderConfig
from satipatthana.configs.factory import create_decoder_config


class ReconstructionDecoder(BaseDecoder):
    """
    Simple MLP Decoder for Input Reconstruction.
    Latent Space (dim) -> Output (input_dim).
    """

    def __init__(self, config: ReconstructionDecoderConfig):
        if isinstance(config, dict):
            config = create_decoder_config(config)
        super().__init__(config)

        output_dim = self.config.input_dim
        # Note: input_dim in config defaults to 10 if not provided, unlike original which raised Error.

        hidden_dim = self.config.decoder_hidden_dim

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
