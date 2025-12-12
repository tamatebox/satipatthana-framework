import torch
import torch.nn as nn
from typing import Dict, Any
from satipatthana.components.decoders.base import BaseDecoder
from satipatthana.configs.decoders import CnnDecoderConfig
from satipatthana.configs.factory import create_decoder_config


class CnnDecoder(BaseDecoder):
    """
    CNN Decoder for Vision tasks (Image Reconstruction).
    Converts Latent Vector (Batch, Dim) -> Image (Batch, C, H, W).
    """

    def __init__(self, config: CnnDecoderConfig):
        if isinstance(config, dict):
            config = create_decoder_config(config)
        super().__init__(config)
        self.channels = self.config.channels
        self.img_size = self.config.img_size

        feature_map_size = self.img_size // 16
        hidden_dim = 256 * feature_map_size * feature_map_size

        # Use Upsample + Conv instead of ConvTranspose2d to avoid checkerboard artifacts
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.Unflatten(1, (256, feature_map_size, feature_map_size)),
            # Block 1: 256 -> 128
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Block 2: 128 -> 64
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Block 3: 64 -> 32
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Block 4: 32 -> channels
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, self.channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.decoder(s)
