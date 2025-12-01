from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
from samadhi.configs.vitakka import BaseVitakkaConfig
from samadhi.configs.factory import create_vitakka_config


class BaseVitakka(nn.Module, ABC):
    """
    Base Vitakka (Initial Application/Search) Component Interface.

    Searches for "intentions (Probes)" within the input stream to form the initial state S0.
    This component is responsible for coarse-grained thinking. Handles both Hard and Soft attention modes.
    """

    def __init__(self, config: BaseVitakkaConfig):
        super().__init__()

        if isinstance(config, dict):
            config = create_vitakka_config(config)

        self.config = config

        self.dim = self.config.dim
        self.n_probes = self.config.n_probes

        # Probe (Concepts) Definition
        self.probes = nn.Parameter(torch.randn(self.n_probes, self.dim))
        self.probes.requires_grad = self.config.probe_trainable
        self._normalize_probes()

    def _normalize_probes(self):
        """L2 normalizes the probes."""
        with torch.no_grad():
            self.probes.div_(torch.norm(self.probes, dim=1, keepdim=True))

    def load_probes(self, pretrained_probes: torch.Tensor):
        """
        Loads probes from an external source.
        Delegate probe loading to Vitakka.
        """
        if pretrained_probes.shape != self.probes.shape:
            raise ValueError(f"Shape mismatch: expected {self.probes.shape}, got {pretrained_probes.shape}")
        with torch.no_grad():
            self.probes.copy_(pretrained_probes)
            self._normalize_probes()

    @abstractmethod
    def forward(self, z_adapted: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Vitakka Process: Search & Select.

        Args:
            z_adapted: Adapted input tensor from an Adapter (Batch, Dim)

        Returns:
            s0: (Batch, Dim) - Initial State
            metadata: Log info (winner, confidence, etc.)
        """
        pass
