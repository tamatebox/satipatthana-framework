from typing import Dict, Tuple, Any
import torch
import torch.nn.functional as F

from samadhi.components.vitakka.base import BaseVitakka
from samadhi.configs.vitakka import StandardVitakkaConfig
from samadhi.configs.factory import create_vitakka_config
from samadhi.utils.logger import get_logger

logger = get_logger(__name__)


class StandardVitakka(BaseVitakka):
    """
    Standard Vitakka Implementation.

    Handles both Hard and Soft attention modes within this single class.
    """

    def __init__(self, config: StandardVitakkaConfig):
        if isinstance(config, dict):
            config = create_vitakka_config(config)
        super().__init__(config)

    def _generate_hard_s0(
        self, z_adapted: torch.Tensor, probs: torch.Tensor, raw_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """[Hard Mode] Selects a single winning probe and performs a clear gate decision."""
        # Max Score Selection
        max_raw_score, winner_idx = torch.max(raw_scores, dim=1)
        is_gate_open = max_raw_score > self.config.gate_threshold

        # Winner Probe
        winner_probes = self.probes[winner_idx]

        # Mix Input & Probe
        alpha = self.config.mix_alpha
        s0_candidate = alpha * z_adapted + (1 - alpha) * winner_probes

        # Gate Application
        gate_mask = is_gate_open.float().unsqueeze(1)
        s0 = s0_candidate * gate_mask

        confidence = probs.gather(1, winner_idx.unsqueeze(1)).squeeze(1)

        return s0, {
            "winner_id": winner_idx,
            "raw_score": max_raw_score,
            "gate_open": is_gate_open,
            "confidence": confidence,
        }

    def _generate_soft_s0(
        self, z_adapted: torch.Tensor, probs: torch.Tensor, raw_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Soft Attention Vitakka.
        [Soft Mode] Computes S0 as a weighted average of probes according to their probability distribution. Suitable for training (differentiable).
        """
        # Weighted Probe Sum
        # (Batch, N) @ (N, Dim) -> (Batch, Dim)
        weighted_probes = torch.matmul(probs, self.probes)

        # Mix Input & Weighted Probe
        alpha = self.config.mix_alpha
        s0_candidate = alpha * z_adapted + (1 - alpha) * weighted_probes

        # Soft Gate Logic (Differentiable)
        # Uses expected score (weighted average score)
        avg_score = torch.sum(raw_scores * probs, dim=1)
        gate_logits = (avg_score - self.config.gate_threshold) * 10.0
        gate_mask = torch.sigmoid(gate_logits).unsqueeze(1)

        s0 = s0_candidate * gate_mask

        # Log Metadata (for consistency, calculate winner as usual)
        winner_idx = torch.argmax(probs, dim=1)
        max_raw_score = torch.max(raw_scores, dim=1)[0]  # Logging uses max
        confidence = torch.max(probs, dim=1)[0]

        return s0, {
            "winner_id": winner_idx,
            "raw_score": max_raw_score,
            "gate_open": max_raw_score > self.config.gate_threshold,  # Logging logic
            "confidence": confidence,
        }

    def forward(self, z_adapted: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Vitakka Process: Search & Select.

        Args:
            z_adapted: Adapted input tensor from an Adapter (Batch, Dim)

        Returns:
            s0: (Batch, Dim) - Initial State
            metadata: Log info (winner, confidence, etc.)
        """
        # 1. Input is already adapted, no need for self.adapter(x_input)
        x_norm = F.normalize(z_adapted, p=2, dim=1)

        # 2. Compute Similarity
        # (Batch, Dim) @ (n_probes, Dim).T -> (Batch, n_probes)
        raw_scores = torch.matmul(x_norm, self.probes.T)

        # 3. Compute Probabilities
        temp = self.config.softmax_temp
        probs = F.softmax(raw_scores / temp, dim=1)

        # 4. Generate S0 (Mode-based switching)
        # Determine mode based on training status
        if self.training:
            mode = self.config.training_attention_mode
        else:
            mode = self.config.prediction_attention_mode

        if mode == "soft":
            s0, partial_meta = self._generate_soft_s0(z_adapted, probs, raw_scores)
        else:
            s0, partial_meta = self._generate_hard_s0(z_adapted, probs, raw_scores)

        # 5. Metadata Construction
        # Converts indices to labels if on CPU/Single item, otherwise keeps tensor logic for batch efficiency.

        metadata = {
            "winner_id": partial_meta["winner_id"],
            "confidence": partial_meta["confidence"],
            "raw_score": partial_meta["raw_score"],
            "gate_open": partial_meta["gate_open"],
            "probs": probs,
            "raw_scores": raw_scores,
        }

        # Logging (Debug Level)
        gate_open_count = (
            partial_meta["gate_open"].float().sum().item()
            if isinstance(partial_meta["gate_open"], torch.Tensor)
            else int(partial_meta["gate_open"])
        )
        batch_size = z_adapted.size(0)
        logger.debug(f"Vitakka Forward: Mode={mode}, Batch={batch_size}, Gates Open={gate_open_count}/{batch_size}")

        return s0, metadata
