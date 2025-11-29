from typing import Dict, Any
import torch.nn as nn
from src.model.samadhi import SamadhiModel  # Import SamadhiModel

# Import components if needed for specific configurations, though SamadhiModel handles instantiation
from src.components.vicara import VicaraBase


class ConvSamadhiModel(SamadhiModel):
    """
    Convolutional Samadhi Model (Latent Samadhi).

    Compresses image data into a low-dimensional latent vector using an Adapter, then performs "search (Vitakka)" and "purification (Vicara)" within that latent space. Finally, it reconstructs an image from the purified latent vector using a Decoder.
    """

    def __init__(self, config):
        # 1. Load image configuration
        self.channels = config.get("channels", 3)
        self.img_size = config.get("img_size", 32)

        # While it's possible to add adapter/decoder info for ConvSamadhi to config before super().__init__,
        # we directly override _build_adapter etc. here.
        super().__init__(config)

        # Replace Vitakka's default Adapter with a CNN-based Adapter.
        self.vitakka.adapter = self._build_cnn_adapter()

    def _build_vicara(self, config: Dict[str, Any]) -> VicaraBase:
        # Vicara's refiner is typically Linear, so no change is required. Override if necessary.
        return super()._build_vicara(config)

    def _build_cnn_adapter(self) -> nn.Module:
        """
        CNN Encoder for Vitakka.
        Image (C, H, W) -> Latent Vector (Dim)
        """
        return nn.Sequential(
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

    def _build_decoder(self) -> nn.Module:
        """
        [Override] CNN Decoder for SamadhiModel.
        Latent Vector (Dim) -> Image (C, H, W)
        """
        feature_map_size = self.img_size // 16
        hidden_dim = 256 * feature_map_size * feature_map_size

        return nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.Unflatten(1, (256, feature_map_size, feature_map_size)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
