from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn

from src.components.refiners.base import BaseRefiner


class BaseVicara(nn.Module, ABC):
    """
    Vicāra (Sustained Application/Refinement) Component Base Class.

    Receives an initial state S0 and purifies (refines) it by removing noise through a recursive process.
    This component is responsible for fine-grained thinking.
    """

    def __init__(self, config: Dict[str, Any], refiners: nn.ModuleList[BaseRefiner]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]
        self.steps = config["refine_steps"]
        self.refiners = refiners

    def forward(
        self, s0: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[float]]:
        """
        Vicara Process: Recursive Refinement.

        Args:
            s0: Initial State (Batch, Dim)
            context: Optional context from Vitakka (e.g., probs for weighted refinement)

        Returns:
            s_final: Converged State (Batch, Dim)
            trajectory: List of state vectors (for visualization)
            energies: List of energy values (stability loss)
        """
        s_t = s0.clone()

        # Logging containers
        trajectory = []
        energies = []

        if not self.training:
            trajectory.append(s_t.detach().cpu().numpy())

        for _ in range(self.steps):
            s_prev = s_t.clone()

            residual = self._refine_step(s_t, context)
            s_t = self.update_state(s_t, residual)

            dist = torch.norm(s_t - s_prev, dim=1)
            energy = dist.mean().item()
            energies.append(energy)

            if not self.training:
                trajectory.append(s_t.detach().cpu().numpy())

            if not self.training and energy < 1e-4:
                break

        return s_t, trajectory, energies

    def update_state(self, s_t: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Updates the state using Inertial Update (EMA).
        s_new = alpha * s_old + (1 - alpha) * residual
        """
        alpha = self.config.get("inertia", 0.7)
        return alpha * s_t + (1 - alpha) * residual

    @abstractmethod
    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Executes a single step of purification calculation.
        """
        pass
