from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any
from satipatthana.configs.decoders import BaseDecoderConfig
from satipatthana.configs.factory import create_decoder_config


class BaseDecoder(nn.Module, ABC):
    """
    Base Decoder Interface.
    Responsible for converting the purified Satipatthana latent state (s_final)
    into the target output format (reconstruction, class logits, etc.).
    """

    def __init__(self, config: BaseDecoderConfig):
        super().__init__()

        if isinstance(config, dict):
            config = create_decoder_config(config)

        self.config = config

        self.dim = self.config.dim

    @abstractmethod
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Purified latent state (Batch, Dim)
        Returns:
            output: Target output (Batch, OutputDim) or Logits
        """
        pass
