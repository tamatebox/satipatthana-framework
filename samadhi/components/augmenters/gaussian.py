"""
GaussianNoiseAugmenter: Adds Gaussian noise proportional to noise_level.
"""

import torch
from typing import Tuple

from samadhi.components.augmenters.base import BaseAugmenter
from samadhi.configs.augmenter import GaussianNoiseAugmenterConfig


class GaussianNoiseAugmenter(BaseAugmenter):
    """
    Gaussian Noise Augmenter.

    Adds Gaussian noise to input proportional to noise_level.
    Severity is computed as: noise_level * max_noise_std
    """

    def __init__(self, config: GaussianNoiseAugmenterConfig = None):
        if config is None:
            config = GaussianNoiseAugmenterConfig()
        super().__init__(config)

    def forward(
        self, x: torch.Tensor, noise_level: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Gaussian noise to input.

        Args:
            x: Raw input tensor (Batch, *)
            noise_level: Noise intensity (0.0 = no noise, 1.0 = maximum noise)

        Returns:
            x_augmented: Noisy tensor with same shape as x
            severity: Per-sample noise intensity tensor (Batch,)
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        # Calculate severity based on noise_level
        actual_std = noise_level * self.config.max_noise_std
        severity = torch.full((batch_size,), actual_std, device=device, dtype=dtype)

        # No noise case - return clone
        if noise_level == 0.0 or actual_std == 0.0:
            return x.clone(), severity

        # Add Gaussian noise
        noise = torch.randn_like(x) * actual_std
        x_augmented = x + noise

        return x_augmented, severity
