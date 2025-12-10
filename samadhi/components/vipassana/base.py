"""
BaseVipassana: Abstract base class for introspection/meta-cognition.

Vipassana (Insight) analyzes the thinking process (SantanaLog) to
determine the quality and confidence of the Samatha convergence.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn

from samadhi.core.santana import SantanaLog
from samadhi.configs.vipassana import BaseVipassanaConfig
from samadhi.utils.logger import get_logger

logger = get_logger(__name__)


class BaseVipassana(nn.Module, ABC):
    """
    Base Vipassana (Insight/Meta-cognition) Interface.

    Analyzes the converged state (S*) and thinking trajectory (SantanaLog)
    to produce:
        - Context vector (V_ctx): Embedding of "doubt/ambiguity"
        - Trust score (alpha): Scalar confidence score (0.0-1.0)

    The context vector can be used by the ConditionalDecoder to produce
    "humble" outputs that reflect uncertainty in the thinking process.
    """

    def __init__(self, config: BaseVipassanaConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, s_star: torch.Tensor, santana: SantanaLog
    ) -> Tuple[torch.Tensor, float]:
        """
        Analyze the thinking process and produce confidence metrics.

        Args:
            s_star: Converged state tensor (Batch, Dim)
            santana: SantanaLog containing the thinking trajectory

        Returns:
            v_ctx: Context vector (Batch, context_dim) - embedding of "doubt"
            trust_score: Scalar confidence score (0.0-1.0) for external control
        """
        pass
