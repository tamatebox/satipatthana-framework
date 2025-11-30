from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseDecoder(nn.Module, ABC):
    """
    Base Decoder Interface.
    Responsible for converting the purified Samadhi latent state (s_final)
    into the target output format (reconstruction, class logits, etc.).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]

    @abstractmethod
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Purified latent state (Batch, Dim)
        Returns:
            output: Target output (Batch, OutputDim) or Logits
        """
        pass
