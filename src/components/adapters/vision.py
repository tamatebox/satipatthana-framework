import torch
import torch.nn as nn
from typing import Dict, Any
from src.components.adapters.base import BaseAdapter


class CnnAdapter(BaseAdapter):
    """
    CNN Adapter for Vision tasks.
    Converts Image (Batch, C, H, W) -> Latent Vector (Batch, Dim).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.channels = config.get("channels", 3)
        self.img_size = config.get("img_size", 32)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * (self.img_size // 16) * (self.img_size // 16), self.dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
