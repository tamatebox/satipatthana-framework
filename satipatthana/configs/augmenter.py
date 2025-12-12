"""
Augmenter Configuration classes for Satipatthana Framework v4.0.
"""

from dataclasses import dataclass
from typing import Optional
from satipatthana.configs.base import BaseConfig
from satipatthana.configs.enums import AugmenterType


@dataclass
class BaseAugmenterConfig(BaseConfig):
    """Base configuration for all Augmenters."""

    type: AugmenterType = AugmenterType.IDENTITY


@dataclass
class IdentityAugmenterConfig(BaseAugmenterConfig):
    """Configuration for Identity Augmenter (no augmentation)."""

    type: AugmenterType = AugmenterType.IDENTITY


@dataclass
class GaussianNoiseAugmenterConfig(BaseAugmenterConfig):
    """Configuration for Gaussian Noise Augmenter."""

    type: AugmenterType = AugmenterType.GAUSSIAN_NOISE
    max_noise_std: float = 0.1  # Maximum standard deviation of noise

    def validate(self):
        if self.max_noise_std < 0:
            raise ValueError("max_noise_std must be non-negative")
