"""
Vipassana Configuration classes for Satipatthana Framework v4.0.
"""

from dataclasses import dataclass
from satipatthana.configs.base import BaseConfig
from satipatthana.configs.enums import VipassanaType


@dataclass
class BaseVipassanaConfig(BaseConfig):
    """Base configuration for all Vipassana components."""

    type: VipassanaType = VipassanaType.STANDARD
    context_dim: int = 32  # Dimension of context vector V_ctx


@dataclass
class StandardVipassanaConfig(BaseVipassanaConfig):
    """
    Configuration for Standard Vipassana.

    Uses a simple encoder to compress the trajectory into context.
    """

    type: VipassanaType = VipassanaType.STANDARD
    hidden_dim: int = 64  # Hidden dimension for encoder


@dataclass
class LSTMVipassanaConfig(BaseVipassanaConfig):
    """
    Configuration for LSTM-based Vipassana.

    Uses LSTM to encode variable-length trajectories.
    """

    type: VipassanaType = VipassanaType.LSTM
    hidden_dim: int = 64
    num_layers: int = 1
    bidirectional: bool = False
