"""
SimpleAuxHead: Auxiliary head for Stage 1 label guidance.

Simple MLP that predicts task output directly from the converged state S*.
Used optionally during training to provide gradient signal for convergence.
"""

import torch
import torch.nn as nn

from samadhi.components.decoders.base import BaseDecoder
from samadhi.configs.decoders import SimpleAuxHeadConfig
from samadhi.configs.factory import create_decoder_config


class SimpleAuxHead(BaseDecoder):
    """
    Simple Auxiliary Head for label guidance.

    Input: S* (converged state)
    Output: Task prediction (e.g., class logits)

    Used only during Stage 1 training (if use_label_guidance=True)
    to provide direct supervision signal to guide convergence.
    """

    def __init__(self, config: SimpleAuxHeadConfig):
        if isinstance(config, dict):
            config = create_decoder_config(config)
        super().__init__(config)

        self.output_dim = config.output_dim
        hidden_dim = config.decoder_hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from converged state.

        Args:
            s: Converged state S* (Batch, dim)

        Returns:
            output: Task prediction (Batch, output_dim)
        """
        return self.net(s)
