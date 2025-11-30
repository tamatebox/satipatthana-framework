from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseRefiner(nn.Module, ABC):
    """
    Base Refiner Interface.
    Represents the mathematical transformation \\Phi(s_t) used inside Vicara.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]

    @abstractmethod
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Latent state (Batch, Dim)
        Returns:
            residual: Transformation result (Batch, Dim) - usually added to s
        """
        pass
