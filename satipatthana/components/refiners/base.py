from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from satipatthana.configs.base import BaseConfig


class BaseRefiner(nn.Module, ABC):
    """
    Base Refiner Interface.
    Perform single step of purification: s_t -> residual
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        if isinstance(config, dict):
            self.dim = config["dim"]
        else:
            self.dim = config.dim

    @abstractmethod
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Current state (Batch, Dim)
        Returns:
            residual: Proposed update vector (Batch, Dim) OR New State
            (Usually Refiner predicts residual or next state, handled by Vicara update_state)
        """
        pass
