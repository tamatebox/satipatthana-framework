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

    Uses GRU to encode trajectory dynamics and MLP to project grounding metrics.
    The context vector V_ctx is the concatenation of both branches.

    Triple Score System:
        - trust_score: Based on static metrics (OOD detection, result-based)
        - conformity_score: Based on dynamic_context (pattern conformity, process-based)
        - confidence_score: Based on both (comprehensive assessment)

    Attributes:
        latent_dim: Dimension of input S* state
        gru_hidden_dim: Hidden dimension for trajectory GRU encoder (Dynamic Context)
        metric_proj_dim: Projection dimension for 8 grounding metrics (Static Context)
        max_steps: Maximum refinement steps (for normalizing convergence_steps)
        context_dim: Auto-computed as gru_hidden_dim + metric_proj_dim
        trust_weight: Loss weight for trust_score (default 1.0)
        conformity_weight: Loss weight for conformity_score (default 1.0)
        confidence_weight: Loss weight for confidence_score (default 1.0)
    """

    type: VipassanaType = VipassanaType.STANDARD
    latent_dim: int = 64
    gru_hidden_dim: int = 32
    metric_proj_dim: int = 32
    max_steps: int = 10
    context_dim: int = 64  # Default, will be overwritten in __post_init__

    # Triple Score weights for Stage 2 loss
    trust_weight: float = 1.0
    conformity_weight: float = 1.0
    confidence_weight: float = 1.0

    def __post_init__(self):
        """Auto-compute context_dim from component dimensions."""
        self.context_dim = self.gru_hidden_dim + self.metric_proj_dim


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
