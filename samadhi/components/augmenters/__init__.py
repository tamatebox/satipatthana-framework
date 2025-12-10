"""
Augmenters: Input transformation components for Samadhi Framework v4.0.

Augmenters apply environmental noise to raw input data, simulating
real-world data quality variations for robust training.
"""

from samadhi.components.augmenters.base import BaseAugmenter
from samadhi.components.augmenters.identity import IdentityAugmenter
from samadhi.components.augmenters.gaussian import GaussianNoiseAugmenter

__all__ = ["BaseAugmenter", "IdentityAugmenter", "GaussianNoiseAugmenter"]
