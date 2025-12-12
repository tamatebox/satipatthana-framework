"""
Augmenters: Input transformation components for Satipatthana Framework v4.0.

Augmenters apply environmental noise to raw input data, simulating
real-world data quality variations for robust training.
"""

from satipatthana.components.augmenters.base import BaseAugmenter
from satipatthana.components.augmenters.identity import IdentityAugmenter
from satipatthana.components.augmenters.gaussian import GaussianNoiseAugmenter

__all__ = ["BaseAugmenter", "IdentityAugmenter", "GaussianNoiseAugmenter"]
