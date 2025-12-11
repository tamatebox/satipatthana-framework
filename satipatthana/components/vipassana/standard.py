"""
StandardVipassana: GRU-based trajectory encoder with grounding metrics.

Encodes the thinking process (trajectory) using a GRU for dynamic context,
and projects 8 grounding metrics for static context. The two are fused
to produce V_ctx and trust_score.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from satipatthana.components.vipassana.base import BaseVipassana, VipassanaOutput
from satipatthana.core.santana import SantanaLog
from satipatthana.configs.vipassana import StandardVipassanaConfig


class StandardVipassana(BaseVipassana):
    """
    Standard Vipassana with GRU trajectory encoder and grounding metrics fusion.

    Architecture:
        Branch 1 (Dynamic): GRU encodes variable-length trajectory -> dynamic context
        Branch 2 (Static): 8 metrics -> MLP projection -> static context
        Fusion: Concat(dynamic, static) -> V_ctx

    8 Grounding Metrics:
        1. velocity: ||S_T - S_{T-1}|| (final movement)
        2. avg_energy: mean(||S_t - S_{t-1}||^2) over valid steps
        3. convergence_steps: t / max_steps (normalized thinking time)
        4. min_dist: min(||S* - P||) (familiarity)
        5. entropy: probe distribution entropy (ambiguity)
        6. s0_min_dist: min(||S0 - P||) (initial OOD degree)
        7. drift_magnitude: ||S* - S0|| (total movement)
        8. recon_error: reconstruction loss (reality check)
    """

    NUM_METRICS = 8

    def __init__(self, config: StandardVipassanaConfig = None):
        if config is None:
            config = StandardVipassanaConfig()
        super().__init__(config)

        self.latent_dim = config.latent_dim
        self.gru_hidden_dim = config.gru_hidden_dim
        self.metric_proj_dim = config.metric_proj_dim
        self.max_steps = config.max_steps

        self._build_networks()

    def _build_networks(self):
        """Build encoder networks."""
        # Branch 1: Dynamic Context - GRU for trajectory encoding
        self.trajectory_gru = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=False,  # Input: (Steps, Batch, Dim)
        )

        # Branch 2: Static Context - MLP for metrics projection
        self.metric_projector = nn.Sequential(
            nn.Linear(self.NUM_METRICS, self.metric_proj_dim * 2),
            nn.ReLU(),
            nn.Linear(self.metric_proj_dim * 2, self.metric_proj_dim),
        )

        # Triple Score Heads
        # 1. Trust head: metrics -> trust_score (OOD detection, result-based)
        #    No gradient to GRU
        self.trust_head = nn.Sequential(
            nn.Linear(self.NUM_METRICS, 2 * self.NUM_METRICS),
            nn.ReLU(),
            nn.Linear(2 * self.NUM_METRICS, 1),
            nn.Sigmoid(),
        )

        # 2. Conformity head: dynamic_context -> conformity_score (pattern conformity, process-based)
        #    Provides gradient to GRU
        self.conformity_head = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gru_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 3. Confidence head: dynamic_context + metrics -> confidence_score (comprehensive)
        #    Provides gradient to GRU
        self.confidence_head = nn.Sequential(
            nn.Linear(self.gru_hidden_dim + self.NUM_METRICS, self.gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gru_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _compute_pairwise_distances(self, points: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances (MPS-compatible).

        Args:
            points: Points tensor (Batch, Dim)
            probes: Probe vectors (N_probes, Dim)

        Returns:
            dists: Pairwise distances (Batch, N_probes)
        """
        points_sq = (points**2).sum(dim=1, keepdim=True)
        probes_sq = (probes**2).sum(dim=1, keepdim=True).T
        cross_term = torch.mm(points, probes.T)
        dists_sq = points_sq + probes_sq - 2 * cross_term
        return torch.sqrt(dists_sq.clamp(min=1e-9))

    def _compile_metrics(
        self,
        s_star: torch.Tensor,
        santana: SantanaLog,
        probes: Optional[torch.Tensor],
        recon_error: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute and normalize all 8 grounding metrics with mask support.

        Args:
            s_star: Converged state (Batch, Dim)
            santana: Trajectory log with convergence_steps
            probes: Probe vectors (N_probes, Dim) or None
            recon_error: Per-sample reconstruction error (Batch, 1) or None

        Returns:
            metrics: Normalized metrics tensor (Batch, 8)
        """
        batch_size = s_star.size(0)
        device = s_star.device
        dtype = s_star.dtype

        trajectory, lengths = santana.get_padded_trajectory()
        # trajectory: (Steps, Batch, Dim), lengths: (Batch,) CPU

        s0 = santana.get_initial_state()
        if s0 is None:
            s0 = s_star

        # Handle empty trajectory case
        if trajectory.numel() == 0:
            num_steps = 0
            lengths_device = torch.ones(batch_size, device=device)  # Fallback: 1 step
        else:
            num_steps = trajectory.size(0)
            lengths_device = lengths.to(device).float()

        # 1. velocity: ||S* - S_{T-1}|| (final movement)
        if num_steps >= 2:
            # Use s_star (converged state) and the state before it
            # prev_indices = lengths - 2 (0-indexed), clamped to valid range
            trajectory_device = trajectory.to(device)
            prev_indices = (lengths_device - 2).clamp(min=0).long()

            # trajectory: (Steps, Batch, Dim) -> (Batch, Steps, Dim) for gather
            traj_batch_first = trajectory_device.permute(1, 0, 2)

            # Gather S_{T-1} for each sample using their individual prev_indices
            batch_indices = torch.arange(batch_size, device=device)
            prev_states = traj_batch_first[batch_indices, prev_indices]  # (Batch, Dim)

            # velocity = ||S* - S_{T-1}||
            final_diff = s_star - prev_states
            velocity = torch.norm(final_diff, dim=1, keepdim=True)
        else:
            velocity = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # 2. avg_energy: masked average of ||S_t - S_{t-1}||^2
        if num_steps >= 2:
            trajectory_device = trajectory.to(device)
            state_diffs = trajectory_device[1:] - trajectory_device[:-1]  # (Steps-1, Batch, Dim)
            step_energies = (state_diffs**2).sum(dim=2)  # (Steps-1, Batch)

            # Create mask for valid steps: step < convergence_steps - 1
            step_indices = torch.arange(num_steps - 1, device=device).unsqueeze(1)  # (Steps-1, 1)
            valid_mask = step_indices < (lengths_device.unsqueeze(0) - 1)  # (Steps-1, Batch)

            # Masked sum and count
            masked_energies = step_energies * valid_mask.float()
            valid_counts = valid_mask.float().sum(dim=0).clamp(min=1)  # (Batch,)
            avg_energy = (masked_energies.sum(dim=0) / valid_counts).unsqueeze(1)  # (Batch, 1)
        else:
            avg_energy = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # 3. convergence_steps: t / max_steps (linear normalization)
        convergence_steps_norm = (lengths_device / self.max_steps).unsqueeze(1)  # (Batch, 1)

        # 4. min_dist: min(||S* - P||) (familiarity)
        # 5. entropy: probe distribution entropy (ambiguity)
        if probes is not None:
            dists = self._compute_pairwise_distances(s_star, probes)
            min_dist, _ = torch.min(dists, dim=1, keepdim=True)
            probs = F.softmax(-dists, dim=1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True)
        else:
            min_dist = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            entropy = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # 6. s0_min_dist: min(||S0 - P||) (initial OOD degree)
        if probes is not None:
            s0_dists = self._compute_pairwise_distances(s0, probes)
            s0_min_dist, _ = torch.min(s0_dists, dim=1, keepdim=True)
        else:
            s0_min_dist = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # 7. drift_magnitude: ||S* - S0||
        drift_magnitude = torch.norm(s_star - s0, dim=1, keepdim=True)

        # 8. recon_error
        if recon_error is None:
            recon_error = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # Apply normalization: log1p for unbounded, none for bounded
        metrics = torch.cat(
            [
                torch.log1p(velocity),  # 1. velocity
                torch.log1p(avg_energy),  # 2. avg_energy
                convergence_steps_norm,  # 3. convergence_steps (already 0-1)
                torch.log1p(min_dist),  # 4. min_dist
                entropy,  # 5. entropy (already bounded)
                torch.log1p(s0_min_dist),  # 6. s0_min_dist
                torch.log1p(drift_magnitude),  # 7. drift_magnitude
                torch.log1p(recon_error),  # 8. recon_error
            ],
            dim=1,
        )  # (Batch, 8)

        return metrics

    def _encode_trajectory(self, santana: SantanaLog, device: torch.device, batch_size: int) -> torch.Tensor:
        """
        Encode trajectory using GRU with pack_padded_sequence for variable-length support.

        Args:
            santana: Trajectory log
            device: Target device
            batch_size: Batch size (for empty trajectory fallback)

        Returns:
            dynamic_context: GRU final hidden state (Batch, gru_hidden_dim)
        """
        trajectory, lengths = santana.get_padded_trajectory()
        # trajectory: (Steps, Batch, Dim), lengths: (Batch,) CPU LongTensor

        if trajectory.numel() == 0:
            # Empty trajectory fallback
            return torch.zeros(batch_size, self.gru_hidden_dim, device=device)

        trajectory = trajectory.to(device)
        batch_size = trajectory.size(1)

        # Ensure lengths are valid (at least 1, at most num_steps)
        num_steps = trajectory.size(0)
        lengths = lengths.clamp(min=1, max=num_steps)

        # Sort by length (descending) for pack_padded_sequence
        lengths_sorted, sort_indices = lengths.sort(descending=True)
        trajectory_sorted = trajectory[:, sort_indices, :]

        # Pack sequence
        packed = pack_padded_sequence(trajectory_sorted, lengths_sorted.cpu(), batch_first=False, enforce_sorted=True)

        # Run GRU
        _, hidden = self.trajectory_gru(packed)  # hidden: (1, Batch, hidden_dim)
        hidden = hidden.squeeze(0)  # (Batch, hidden_dim)

        # Unsort to restore original order
        _, unsort_indices = sort_indices.sort()
        dynamic_context = hidden[unsort_indices]

        return dynamic_context

    def forward(
        self,
        s_star: torch.Tensor,
        santana: SantanaLog,
        probes: Optional[torch.Tensor] = None,
        recon_error: Optional[torch.Tensor] = None,
    ) -> VipassanaOutput:
        """
        Analyze the thinking process and produce context vector and triple scores.

        Triple Score System:
            - trust_score: metrics -> trust (OOD detection, result-based, NO gradient to GRU)
            - conformity_score: dynamic_context -> conformity (pattern conformity, process-based)
            - confidence_score: dynamic_context + metrics -> confidence (comprehensive)

        Args:
            s_star: Converged state tensor (Batch, Dim)
            santana: SantanaLog containing the thinking trajectory
            probes: Probe vectors from Vitakka (N_probes, Dim), optional
            recon_error: Per-sample reconstruction error (Batch, 1), optional

        Returns:
            VipassanaOutput containing:
                - v_ctx: Context vector (Batch, context_dim) - fused dynamic + static context
                - trust_score: Trust score from metrics (Batch, 1)
                - conformity_score: Conformity score from dynamic_context (Batch, 1)
                - confidence_score: Confidence score from both (Batch, 1)
        """
        batch_size = s_star.size(0)
        device = s_star.device

        # Branch 1: Dynamic Context from trajectory GRU
        dynamic_context = self._encode_trajectory(santana, device, batch_size)  # (Batch, gru_hidden_dim)

        # Branch 2: Static Context from grounding metrics
        metrics = self._compile_metrics(s_star, santana, probes, recon_error)  # (Batch, 8)
        static_context = self.metric_projector(metrics)  # (Batch, metric_proj_dim)

        # Fusion: concatenate dynamic and static context
        v_ctx = torch.cat([dynamic_context, static_context], dim=1)  # (Batch, context_dim)

        # Triple Scores
        # 1. Trust score: metrics only (no gradient to GRU)
        trust_score = self.trust_head(metrics)  # (Batch, 1)

        # 2. Conformity score: dynamic_context only (gradient flows to GRU)
        conformity_score = self.conformity_head(dynamic_context)  # (Batch, 1)

        # 3. Confidence score: both dynamic_context and metrics (gradient flows to GRU)
        combined_features = torch.cat([dynamic_context, metrics], dim=1)  # (Batch, gru_hidden_dim + 8)
        confidence_score = self.confidence_head(combined_features)  # (Batch, 1)

        return VipassanaOutput(
            v_ctx=v_ctx,
            trust_score=trust_score,
            conformity_score=conformity_score,
            confidence_score=confidence_score,
        )
