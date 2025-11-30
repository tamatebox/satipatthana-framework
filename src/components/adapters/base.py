from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseAdapter(nn.Module, ABC):
    """
    Base Adapter Interface.
    Responsible for converting raw input into the Samadhi latent space (s0).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw input tensor (Batch, *)
        Returns:
            z: Latent vector (Batch, Dim)
        """
        pass
