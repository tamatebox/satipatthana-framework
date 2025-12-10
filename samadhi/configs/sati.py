"""
Sati Configuration classes for Samadhi Framework v4.0.
"""

from dataclasses import dataclass
from samadhi.configs.base import BaseConfig
from samadhi.configs.enums import SatiType


@dataclass
class BaseSatiConfig(BaseConfig):
    """Base configuration for all Sati components."""

    type: SatiType = SatiType.FIXED_STEP


@dataclass
class FixedStepSatiConfig(BaseSatiConfig):
    """
    Configuration for Fixed Step Sati.

    Always runs for a fixed number of steps without early stopping.
    """

    type: SatiType = SatiType.FIXED_STEP


@dataclass
class ThresholdSatiConfig(BaseSatiConfig):
    """
    Configuration for Threshold-based Sati.

    Stops when energy (state change) falls below threshold.
    """

    type: SatiType = SatiType.THRESHOLD
    energy_threshold: float = 1e-4  # Stop when energy < threshold
    min_steps: int = 1  # Minimum steps before allowing early stop

    def validate(self):
        if self.energy_threshold < 0:
            raise ValueError("energy_threshold must be non-negative")
        if self.min_steps < 1:
            raise ValueError("min_steps must be at least 1")
