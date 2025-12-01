import torch
import torch.nn as nn
from typing import Dict, Any, Union
from samadhi.components.adapters.base import BaseAdapter
from samadhi.configs.adapters import CnnAdapterConfig
from samadhi.configs.factory import create_adapter_config


class CnnAdapter(BaseAdapter):
    """
    CNN Adapter for Vision tasks.
    Converts Image (Batch, C, H, W) -> Latent Vector (Batch, Dim).
    """

    def __init__(self, config: CnnAdapterConfig):
        if isinstance(config, dict):
            if "type" not in config:
                config["type"] = "cnn"
            config = create_adapter_config(config)

        super().__init__(config)

        self.channels = self.config.channels
        self.img_size = self.config.img_size

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
