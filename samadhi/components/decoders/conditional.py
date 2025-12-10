"""
ConditionalDecoder: Task decoder that uses Vipassana context vector.

Takes concatenated input of latent state (S*) and context vector (V_ctx)
to produce task-specific outputs with uncertainty awareness.
"""

import torch
import torch.nn as nn

from samadhi.components.decoders.base import BaseDecoder
from samadhi.configs.decoders import ConditionalDecoderConfig
from samadhi.configs.factory import create_decoder_config


class ConditionalDecoder(BaseDecoder):
    """
    Conditional Decoder for task-specific outputs.

    Input: Concatenation of S* (dim) and V_ctx (context_dim)
    Output: Task-specific output (output_dim)

    The context vector allows the decoder to produce "humble" outputs
    that reflect uncertainty in the thinking process.
    """

    def __init__(self, config: ConditionalDecoderConfig):
        if isinstance(config, dict):
            config = create_decoder_config(config)
        super().__init__(config)

        self.context_dim = config.context_dim
        self.output_dim = config.output_dim
        hidden_dim = config.decoder_hidden_dim

        # Input is concatenation of S* and V_ctx
        input_dim = self.dim + self.context_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, s_and_ctx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with concatenated input.

        Args:
            s_and_ctx: Concatenation of S* and V_ctx (Batch, dim + context_dim)

        Returns:
            output: Task-specific output (Batch, output_dim)
        """
        return self.net(s_and_ctx)

    def forward_with_concat(
        self, s_star: torch.Tensor, v_ctx: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method that handles concatenation.

        Args:
            s_star: Converged state (Batch, dim)
            v_ctx: Vipassana context vector (Batch, context_dim)

        Returns:
            output: Task-specific output (Batch, output_dim)
        """
        s_and_ctx = torch.cat([s_star, v_ctx], dim=1)
        return self.forward(s_and_ctx)
