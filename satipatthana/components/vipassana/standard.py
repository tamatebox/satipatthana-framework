"""
StandardVipassana: Simple trajectory encoder with confidence monitoring.

Uses mean/variance aggregation across the trajectory to produce
a context vector and trust score.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

from satipatthana.components.vipassana.base import BaseVipassana
from satipatthana.core.santana import SantanaLog
from satipatthana.configs.vipassana import StandardVipassanaConfig


class StandardVipassana(BaseVipassana):
    """
    Standard Vipassana using mean/variance trajectory aggregation.

    Extracts features from the convergence trajectory:
    - Position: Final converged state
    - Velocity: Movement from initial to final state
    - Smoothness: Inverse of energy variance (lower = smoother)

    The trust score reflects:
    - Fewer steps = higher confidence (quick convergence)
    - Lower total energy = higher confidence (smooth trajectory)
    """

    def __init__(self, config: StandardVipassanaConfig = None):
        if config is None:
            config = StandardVipassanaConfig()
        super().__init__(config)

        self.hidden_dim = config.hidden_dim
        self.context_dim = config.context_dim

        # Will be initialized on first forward pass when dim is known
        self._encoder = None
        self._trust_head = None
        self._input_dim = None

    def _build_networks(self, state_dim: int):
        """Build encoder networks once state dimension is known."""
        # Feature vector: [s_star (dim), velocity_norm (1), avg_energy (1)]
        feature_dim = state_dim + 2
        self._input_dim = state_dim

        self._encoder = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.context_dim),
        )

        self._trust_head = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, s_star: torch.Tensor, santana: SantanaLog) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze the thinking process and produce context vector and trust score.

        Args:
            s_star: Converged state tensor (Batch, Dim)
            santana: SantanaLog containing the thinking trajectory

        Returns:
            v_ctx: Context vector (Batch, context_dim) - embedding of "doubt"
            trust_score: Confidence tensor (Batch, 1) for external control
        """
        batch_size, state_dim = s_star.shape
        device = s_star.device
        dtype = s_star.dtype

        # Lazy initialization of networks
        if self._encoder is None or self._input_dim != state_dim:
            self._build_networks(state_dim)
            self._encoder = self._encoder.to(device)
            self._trust_head = self._trust_head.to(device)

        # Extract trajectory features
        num_steps = len(santana)
        initial_state = santana.get_initial_state()
        total_energy = santana.get_total_energy()

        # Compute velocity (distance from initial to final state)
        if initial_state is not None:
            velocity = torch.norm(s_star - initial_state, dim=1, keepdim=True)
        else:
            velocity = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # Compute average energy
        if num_steps > 0:
            avg_energy = total_energy / num_steps
        else:
            avg_energy = 0.0

        avg_energy_tensor = torch.full((batch_size, 1), avg_energy, device=device, dtype=dtype)

        # Build feature vector: [s_star, velocity_norm, avg_energy]
        features = torch.cat([s_star, velocity, avg_energy_tensor], dim=1)

        # Encode to context vector
        v_ctx = self._encoder(features)

        # Compute trust score (batch-wise tensor)
        trust_score = self._trust_head(features)  # (Batch, 1)

        return v_ctx, trust_score
