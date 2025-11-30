import torch.nn as nn
import torch
from typing import Dict, Any
from src.components.refiners.base import BaseRefiner


class MlpRefiner(BaseRefiner):
    """
    Standard MLP Refiner.
    Phi(s) = Linear -> Norm -> ReLU -> Linear -> Tanh
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Standard bottleneck architecture for purification
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim),
            nn.Tanh(),  # State is bound to -1~1
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)
