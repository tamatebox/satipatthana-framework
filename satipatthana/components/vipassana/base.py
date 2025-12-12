"""
BaseVipassana: Abstract base class for introspection/meta-cognition.

Vipassana (Insight) analyzes the thinking process (SantanaLog) to
determine the quality and confidence of the Samatha convergence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn

from satipatthana.core.santana import SantanaLog
from satipatthana.configs.vipassana import BaseVipassanaConfig
from satipatthana.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VipassanaOutput:
    """
    Output container for Vipassana forward pass.

    Triple Score System:
        - trust_score: Based on static metrics (OOD detection, result-based)
        - conformity_score: Based on dynamic_context (pattern conformity, process-based)
        - confidence_score: Based on both (comprehensive assessment)

    Attributes:
        v_ctx: Context vector (Batch, context_dim) - fused dynamic + static context
        trust_score: Trust score from metrics (Batch, 1)
        conformity_score: Conformity score from dynamic_context (Batch, 1)
        confidence_score: Confidence score from both (Batch, 1)
    """

    v_ctx: torch.Tensor
    trust_score: torch.Tensor
    conformity_score: torch.Tensor
    confidence_score: torch.Tensor


class BaseVipassana(nn.Module, ABC):
    """
    Base Vipassana (Insight/Meta-cognition) Interface.

    Analyzes the converged state (S*) and thinking trajectory (SantanaLog)
    to produce:
        - Context vector (V_ctx): Embedding of "doubt/ambiguity"
        - Triple Scores: trust, conformity, and confidence scores

    The context vector can be used by the ConditionalDecoder to produce
    "humble" outputs that reflect uncertainty in the thinking process.
    """

    def __init__(self, config: BaseVipassanaConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, s_star: torch.Tensor, santana: SantanaLog) -> VipassanaOutput:
        """
        Analyze the thinking process and produce confidence metrics.

        Args:
            s_star: Converged state tensor (Batch, Dim)
            santana: SantanaLog containing the thinking trajectory

        Returns:
            VipassanaOutput containing:
                - v_ctx: Context vector (Batch, context_dim) - embedding of "doubt"
                - trust_score: Trust score (Batch, 1)
                - conformity_score: Conformity score (Batch, 1)
                - confidence_score: Confidence score (Batch, 1)
        """
        pass
