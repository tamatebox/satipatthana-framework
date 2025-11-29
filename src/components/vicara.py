from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VicaraBase(nn.Module, ABC):
    """
    Vicāra (Sustained Application/Refinement) Component Base Class.

    Receives an initial state S0 and purifies (refines) it by removing noise through a recursive process.
    This component is responsible for fine-grained thinking.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]
        self.steps = config["refine_steps"]

        # Refiner Networks (The "Phi" loop)
        # S_t -> S_{t+1} (residual)
        # Standardizes to ModuleList for all subclasses.
        # The default implementation creates a single refiner.
        self.refiners = nn.ModuleList([self._build_refiner()])

    def _build_refiner(self) -> nn.Module:
        """
        Build a single refiner network.
        Can be overridden by subclasses.
        """
        return nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim),
            nn.Tanh(),  # State is bound to -1~1
        )

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
        # Note: Trajectory logging might be heavy for large batches, usually used for inference/debug
        trajectory = []
        energies = []

        # Initial state log (detach for cpu/numpy)
        if not self.training:
            trajectory.append(s_t.detach().cpu().numpy())

        for _ in range(self.steps):
            s_prev = s_t.clone()

            # Refinement Step (Abstracted for Standard vs Weighted)
            residual = self._refine_step(s_t, context)

            # Inertial Update (EMA)
            # s_new = 0.7 * s_old + 0.3 * residual
            s_t = 0.7 * s_t + 0.3 * residual

            # Compute Energy (Stability)
            # Calculates batch-wise mean energy for simple logging, though individual energies could be kept.
            dist = torch.norm(s_t - s_prev, dim=1)
            energy = dist.mean().item()
            energies.append(energy)

            if not self.training:
                trajectory.append(s_t.detach().cpu().numpy())

            # Early Stopping (Appanā) - Inference only
            if not self.training and energy < 1e-4:
                break

        return s_t, trajectory, energies

    def refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Public API for executing a single refinement step.
        Delegates to the subclass-specific _refine_step implementation.
        """
        return self._refine_step(s_t, context)

    @abstractmethod
    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Executes a single step of purification calculation.
        """
        pass


class StandardVicara(VicaraBase):
    """
    Standard Vicāra.
    Performs purification using a single refiner network.
    """

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        # Use the single shared refiner
        return self.refiners[0](s_t)


class WeightedVicara(VicaraBase):
    """
    Weighted Vicāra (Optional / Advanced).
    """

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        # Future extension:
        # if self.multiple_refiners:
        #    return weighted_sum(refiners(s_t), context['probs'])
        return self.refiners[0](s_t)


class ProbeVicara(VicaraBase):
    """
    Probe-Specific Vicāra.
    Possesses a distinct refiner (purification logic) for each probe (concept).
    """

    def __init__(self, config: Dict[str, Any]):
        # Base init creates a single refiner in self.refiners
        super().__init__(config)

        # Overrides self.refiners with n_probes specific refiners.
        # Note: This discards the one created in super().__init__.
        self.n_probes = config["n_probes"]
        self.refiners = nn.ModuleList([self._build_refiner() for _ in range(self.n_probes)])

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        mode = self.config.get("attention_mode", "hard")

        if mode == "soft":
            # Soft Mode: Weighted sum of all refiner outputs based on probe probabilities.
            if "probs" not in context:
                raise ValueError("ProbeVicara in soft mode requires 'probs' in context.")

            probs = context["probs"]
            output = torch.zeros_like(s_t)

            for i, refiner in enumerate(self.refiners):
                # refiner output: (Batch, Dim)
                # weight: (Batch, 1)
                w = probs[:, i].unsqueeze(1)
                output += w * refiner(s_t)

            return output

        else:
            # Hard Mode: Only applies the refiner of the winning probe for each sample.
            winner_ids = context["winner_id"]

            # Case for batch size 1 or single integer (e.g., during inference)
            if isinstance(winner_ids, int):
                return self.refiners[winner_ids](s_t)

            if winner_ids.dim() == 0:  # 0-d tensor
                return self.refiners[winner_ids.item()](s_t)

            # Batch processing: Apply refiner by masking for each winner.
            output = torch.zeros_like(s_t)
            # Loop through each probe and compute/fill for samples where that probe is the winner.
            for i, refiner in enumerate(self.refiners):
                mask = winner_ids == i
                if mask.any():
                    output[mask] = refiner(s_t[mask])

            return output
