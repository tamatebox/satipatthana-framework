from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Union
from satipatthana.configs.adapters import BaseAdapterConfig
from satipatthana.configs.factory import create_adapter_config


class BaseAdapter(nn.Module, ABC):
    """
    Base Adapter Interface.
    Responsible for converting raw input into the Satipatthana latent space (s0).
    """

    def __init__(self, config: BaseAdapterConfig):
        super().__init__()

        # Allow legacy dict config
        if isinstance(config, dict):
            config = create_adapter_config(config)

        self.config = config

        self.dim = self.config.dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw input tensor (Batch, *)
        Returns:
            z: Latent vector (Batch, Dim)
        """
        pass
