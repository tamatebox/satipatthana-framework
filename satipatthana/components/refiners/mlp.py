import torch.nn as nn
import torch
from satipatthana.components.refiners.base import BaseRefiner
from satipatthana.configs.base import BaseConfig


class MlpRefiner(BaseRefiner):
    """
    Standard MLP Refiner.
    Phi(s) = Linear -> Norm -> ReLU -> Linear -> Tanh
    """

    def __init__(self, config: BaseConfig):
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
