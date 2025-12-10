"""
BaseAugmenter: Abstract base class for input augmentation.

Augmenters are responsible for applying environmental noise to raw input,
simulating data quality variations (Aleatoric Uncertainty).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
import torch.nn as nn

from samadhi.configs.augmenter import BaseAugmenterConfig
from samadhi.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAugmenter(nn.Module, ABC):
    """
    Base Augmenter Interface.

    Applies environmental noise to raw input data.
    Returns both the augmented input and the severity (noise level) for each sample.

    Note:
        - Augmenters only transform input data
        - They do NOT control internal Samatha components (that's drunk_mode's job)
        - severity is returned as Tensor for batch processing consistency
    """

    def __init__(self, config: BaseAugmenterConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, x: torch.Tensor, noise_level: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to input.

        Args:
            x: Raw input tensor (Batch, *)
            noise_level: Noise intensity (0.0 = no noise, 1.0 = maximum noise)

        Returns:
            x_augmented: Augmented tensor with same shape as x
            severity: Per-sample noise intensity tensor (Batch,)
        """
        pass
