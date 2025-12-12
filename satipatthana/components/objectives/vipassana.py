"""
VipassanaObjective: Contrastive learning objective for Vipassana training (Stage 2).

Trains Vipassana to distinguish between:
- Good trajectories (normal convergence)
- Bad trajectories (augmented, drunk mode, or mismatched)
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VipassanaObjective:
    """
    Objective function for Stage 2 Vipassana training.

    Uses Binary Cross Entropy to train Vipassana to predict triple scores
    that match the ground truth quality of the thinking process.

    Triple Score System:
        - trust_score: Based on static metrics (OOD detection, result-based)
        - conformity_score: Based on dynamic_context (pattern conformity, process-based)
        - confidence_score: Based on both (comprehensive assessment)

    Target values:
    - Good trajectory (clean input, normal Samatha): 1.0
    - Augmented trajectory: 1.0 - severity
    - Drunk mode trajectory: 0.0
    - Mismatched trajectory (shuffled S*/SantanaLog): 0.0
    - Void trajectory (OOD input): 0.0
    """

    def __init__(
        self,
        device: Optional[str] = None,
        trust_weight: float = 1.0,
        conformity_weight: float = 1.0,
        confidence_weight: float = 1.0,
    ):
        """
        Args:
            device: Target device
            trust_weight: Loss weight for trust_score
            conformity_weight: Loss weight for conformity_score
            confidence_weight: Loss weight for confidence_score
        """
        self.device = torch.device(device) if device else self._get_default_device()
        self.trust_weight = trust_weight
        self.conformity_weight = conformity_weight
        self.confidence_weight = confidence_weight

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def compute_loss(
        self,
        trust_scores: torch.Tensor,
        targets: torch.Tensor,
        conformity_scores: torch.Tensor,
        confidence_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute Vipassana training loss with Triple Score.

        Args:
            trust_scores: Predicted trust scores from Vipassana (Batch, 1)
            targets: Ground truth trust targets (Batch, 1) in [0, 1]
            conformity_scores: Conformity scores from dynamic_context (Batch, 1)
            confidence_scores: Confidence scores from both (Batch, 1)

        Returns:
            total_loss: Weighted sum of BCE losses
            loss_components: Dictionary with loss breakdown
        """
        # 1. Trust score BCE loss
        trust_bce = F.binary_cross_entropy(trust_scores, targets)

        # 2. Conformity score BCE loss
        conformity_bce = F.binary_cross_entropy(conformity_scores, targets)

        # 3. Confidence score BCE loss
        confidence_bce = F.binary_cross_entropy(confidence_scores, targets)

        # Total loss: weighted sum
        total_loss = (
            self.trust_weight * trust_bce
            + self.conformity_weight * conformity_bce
            + self.confidence_weight * confidence_bce
        )

        loss_components = {
            "total_loss": total_loss.item(),
            "trust_bce": trust_bce.item(),
            "conformity_bce": conformity_bce.item(),
            "confidence_bce": confidence_bce.item(),
            "trust_pred_mean": trust_scores.mean().item(),
            "conformity_pred_mean": conformity_scores.mean().item(),
            "confidence_pred_mean": confidence_scores.mean().item(),
            "target_mean": targets.mean().item(),
            "trust_accuracy": ((trust_scores > 0.5) == (targets > 0.5)).float().mean().item(),
            "conformity_accuracy": ((conformity_scores > 0.5) == (targets > 0.5)).float().mean().item(),
            "confidence_accuracy": ((confidence_scores > 0.5) == (targets > 0.5)).float().mean().item(),
        }

        return total_loss, loss_components

    def compute_contrastive_loss(
        self,
        good_trust: torch.Tensor,
        bad_trust: torch.Tensor,
        margin: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute contrastive loss between good and bad trajectories.

        Encourages good trajectories to have higher trust than bad ones.

        Args:
            good_trust: Trust scores for good trajectories (Batch, 1)
            bad_trust: Trust scores for bad trajectories (Batch, 1)
            margin: Minimum margin between good and bad trust

        Returns:
            total_loss: Contrastive margin loss
            loss_components: Dictionary with loss breakdown
        """
        # Margin loss: max(0, margin - (good - bad))
        diff = good_trust - bad_trust
        margin_loss = F.relu(margin - diff).mean()

        # BCE targets
        good_target = torch.ones_like(good_trust)
        bad_target = torch.zeros_like(bad_trust)

        good_bce = F.binary_cross_entropy(good_trust, good_target)
        bad_bce = F.binary_cross_entropy(bad_trust, bad_target)
        bce_loss = (good_bce + bad_bce) / 2

        total_loss = bce_loss + margin_loss

        return total_loss, {
            "total_loss": total_loss.item(),
            "bce_loss": bce_loss.item(),
            "margin_loss": margin_loss.item(),
            "good_trust_mean": good_trust.mean().item(),
            "bad_trust_mean": bad_trust.mean().item(),
            "trust_diff_mean": diff.mean().item(),
        }


class GuidanceLoss:
    """
    Guidance Loss for Stage 1 label-guided training.

    Provides supervision signal from AuxiliaryHead to guide Samatha
    convergence toward task-relevant representations.
    """

    def __init__(
        self,
        task_type: str = "classification",
        device: Optional[str] = None,
    ):
        """
        Args:
            task_type: "classification" or "regression"
            device: Target device
        """
        self.task_type = task_type
        self.device = torch.device(device) if device else self._get_default_device()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def compute_loss(
        self,
        aux_output: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute guidance loss from auxiliary head output.

        Args:
            aux_output: Output from AuxiliaryHead (Batch, output_dim)
            targets: Ground truth labels (Batch,) or (Batch, output_dim)

        Returns:
            loss: Guidance loss
            loss_components: Dictionary with loss breakdown
        """
        if self.task_type == "classification":
            # Cross entropy for classification
            if targets.dim() == 1:
                targets = targets.long()
            loss = F.cross_entropy(aux_output, targets)

            # Compute accuracy
            preds = aux_output.argmax(dim=1)
            if targets.dim() > 1:
                target_labels = targets.argmax(dim=1)
            else:
                target_labels = targets
            accuracy = (preds == target_labels).float().mean().item()

            return loss, {
                "guidance_loss": loss.item(),
                "guidance_accuracy": accuracy,
            }

        else:  # regression
            # MSE for regression
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = F.mse_loss(aux_output, targets)

            return loss, {
                "guidance_loss": loss.item(),
                "guidance_mse": loss.item(),
            }


class ProbeDiversityLoss:
    """
    Probe Diversity Loss for Stage 1 Samatha training.

    Encourages probe vectors to be spread out in latent space by penalizing
    high cosine similarity between probe pairs. This prevents mode collapse
    where all samples converge to a single probe.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device) if device else self._get_default_device()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def compute_loss(
        self,
        probes: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute diversity loss to encourage probe spread.

        Args:
            probes: Probe vectors (N_PROBES, dim)

        Returns:
            loss: Mean off-diagonal cosine similarity (lower = more diverse)
            loss_components: Dictionary with loss breakdown
        """
        n_probes = probes.size(0)

        # Normalize probes
        probes_norm = probes / (probes.norm(dim=1, keepdim=True) + 1e-8)

        # Compute pairwise cosine similarity matrix
        similarity_matrix = torch.mm(probes_norm, probes_norm.t())

        # Mask out diagonal (self-similarity = 1.0)
        mask = ~torch.eye(n_probes, dtype=torch.bool, device=probes.device)
        off_diagonal = similarity_matrix[mask]

        # Loss: mean similarity (want to minimize)
        diversity_loss = off_diagonal.mean()

        # Additional stats
        max_sim = off_diagonal.max().item()
        min_sim = off_diagonal.min().item()

        return diversity_loss, {
            "diversity_loss": diversity_loss.item(),
            "probe_sim_mean": diversity_loss.item(),
            "probe_sim_max": max_sim,
            "probe_sim_min": min_sim,
        }


class StabilityLoss:
    """
    Stability Loss for Stage 1 Samatha training.

    Encourages smooth convergence by penalizing large state changes
    during the Vicara refinement loop.

    Uses stability_pair (s_T, s_T_1) with gradients for differentiable loss.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device) if device else self._get_default_device()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def compute_loss(
        self,
        stability_pair: Tuple[torch.Tensor, torch.Tensor],
        santana=None,  # Optional: for logging only
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute stability loss from stability_pair.

        Args:
            stability_pair: (s_T, s_T_1) tuple with gradients from SamathaEngine
            santana: Optional SantanaLog for additional logging info

        Returns:
            loss: Stability loss ||s_T - s_T_1||^2 (differentiable)
            loss_components: Dictionary with loss breakdown
        """
        s_T, s_T_1 = stability_pair

        # Compute differentiable stability loss: ||s_T - s_T_1||^2
        diff = s_T - s_T_1
        stability_loss = torch.norm(diff, dim=1).pow(2).mean()

        # Additional logging info from santana (if provided)
        num_steps = len(santana.energies) if santana and santana.energies else 0
        total_energy = sum(santana.energies) if santana and santana.energies else 0.0
        final_energy = santana.energies[-1] if santana and santana.energies else 0.0

        return stability_loss, {
            "stability_loss": stability_loss.item(),
            "total_energy": total_energy,
            "final_energy": final_energy,
            "num_steps": num_steps,
        }


__all__ = ["VipassanaObjective", "GuidanceLoss", "ProbeDiversityLoss", "StabilityLoss"]
