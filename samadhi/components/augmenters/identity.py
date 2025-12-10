"""
IdentityAugmenter: Pass-through augmenter with no transformation.
"""

import torch
from typing import Tuple

from samadhi.components.augmenters.base import BaseAugmenter
from samadhi.configs.augmenter import IdentityAugmenterConfig


class IdentityAugmenter(BaseAugmenter):
    """
    Identity Augmenter - no transformation applied.

    Always returns input unchanged with zero severity.
    Used when no augmentation is desired or for testing baselines.
    """

    def __init__(self, config: IdentityAugmenterConfig = None):
        if config is None:
            config = IdentityAugmenterConfig()
        super().__init__(config)

    def forward(
        self, x: torch.Tensor, noise_level: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass through input unchanged.

        Args:
            x: Raw input tensor (Batch, *)
            noise_level: Ignored for identity augmenter

        Returns:
            x_augmented: Clone of input tensor (same shape)
            severity: Zero tensor (Batch,)
        """
        batch_size = x.size(0)
        severity = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        return x.clone(), severity
