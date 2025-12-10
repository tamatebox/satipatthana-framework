from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn

from samadhi.components.refiners.base import BaseRefiner
from samadhi.configs.vicara import BaseVicaraConfig
from samadhi.configs.factory import create_vicara_config
from samadhi.utils.logger import get_logger

logger = get_logger(__name__)


class BaseVicara(nn.Module, ABC):
    """
    Vicāra (Sustained Application/Refinement) Component Base Class.

    Receives an initial state S0 and purifies (refines) it by removing noise through a recursive process.
    This component is responsible for fine-grained thinking.

    In v4.0, Vicara is responsible ONLY for single-step state updates.
    Loop control is delegated to SamathaEngine.

    The legacy forward() method with loop is kept for backwards compatibility
    but will be deprecated in favor of step() + SamathaEngine.
    """

    def __init__(self, config: BaseVicaraConfig, refiners: nn.ModuleList[BaseRefiner]):
        super().__init__()

        if isinstance(config, dict):
            config = create_vicara_config(config)

        self.config = config

        self.dim = self.config.dim
        self.steps = self.config.refine_steps
        self.refiners = refiners

    def step(self, s_t: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        """
        Single-step state update (v4.0 interface).

        This is the primary interface for v4.0. Loop control is handled
        by SamathaEngine, which calls this method repeatedly.

        Args:
            s_t: Current state (Batch, Dim)
            context: Optional context from Vitakka
                - For ProbeSpecificVicara: uses probs, winner_id
                - For Standard/WeightedVicara: ignored

        Returns:
            s_{t+1}: Updated state (Batch, Dim)
        """
        residual = self._refine_step(s_t, context)
        return self.update_state(s_t, residual)

    def forward(
        self, s0: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[float]]:
        """
        Vicara Process: Recursive Refinement.

        Note: This method is kept for backwards compatibility.
        In v4.0, prefer using step() with SamathaEngine for loop control.

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

        logger.debug(f"Vicara Loop Start: Max Steps={self.steps}, Batch={s0.size(0)}")

        for step_idx in range(self.steps):
            s_prev = s_t.clone()

            # Use the new step() method internally
            s_t = self.step(s_t, context)

            dist = torch.norm(s_t - s_prev, dim=1)
            energy = dist.mean().item()
            energies.append(energy)

            logger.debug(f"Vicara Step {step_idx+1}/{self.steps}: Energy={energy:.6f}")

            if not self.training:
                trajectory.append(s_t.detach().cpu().numpy())

            if not self.training and energy < 1e-4:
                logger.debug(f"Vicara Converged at Step {step_idx+1}")
                break

        return s_t, trajectory, energies

    def update_state(self, s_t: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Updates the state using Inertial Update (EMA).
        s_new = alpha * s_old + (1 - alpha) * residual
        """
        alpha = self.config.inertia
        return alpha * s_t + (1 - alpha) * residual

    @abstractmethod
    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Executes a single step of purification calculation.
        """
        pass
