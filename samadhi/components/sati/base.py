"""
BaseSati: Abstract base class for convergence monitoring.

Sati (Mindfulness) is responsible for monitoring the state trajectory
and determining when to stop the Vicara refinement loop.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

from samadhi.core.santana import SantanaLog
from samadhi.configs.sati import BaseSatiConfig
from samadhi.utils.logger import get_logger

logger = get_logger(__name__)


class BaseSati(nn.Module, ABC):
    """
    Base Sati (Mindfulness/Gating) Interface.

    Monitors the state trajectory (SantanaLog) and determines when
    the Vicara refinement loop should stop.

    Responsibilities:
        - Evaluate convergence criteria based on state history
        - Signal stop condition to SamathaEngine
        - Reject noise and hallucinations (gating function)
    """

    def __init__(self, config: BaseSatiConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, current_state: torch.Tensor, santana: SantanaLog
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate whether to stop the Vicara loop.

        Args:
            current_state: Current state tensor (Batch, Dim)
            santana: SantanaLog containing state trajectory history

        Returns:
            should_stop: Boolean indicating whether to stop the loop
            info: Dictionary containing additional information
                  (e.g., convergence metrics, reason for stopping)
        """
        pass
