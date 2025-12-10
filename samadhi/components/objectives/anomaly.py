from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from samadhi.components.objectives.unsupervised import UnsupervisedObjective
from samadhi.configs.main import SamadhiConfig


class AnomalyObjective(UnsupervisedObjective):
    """
    Anomaly Detection Objective with Margin Loss (Contrastive Learning).

    Teaches the model to:
    1. Reconstruct "Normal" data well (Attraction).
    2. Fail to reconstruct "Anomaly" data (Repulsion) using a Margin Loss.

    Loss = Loss_Recon(Normal) + Weight * Max(0, Margin - Loss_Recon(Anomaly))
         + Stability + Entropy + Balance
    """

    def __init__(self, config: SamadhiConfig, device: Optional[str] = None):
        super().__init__(config, device)
        # Use reduction='none' to calculate per-sample loss for masking
        self.recon_loss_fn_none = nn.MSELoss(reduction="none")

        # Cache config values for easier access
        self.margin = self.config.objective.anomaly_margin
        self.anomaly_weight = self.config.objective.anomaly_weight

    def compute_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        s0: torch.Tensor,
        s_final: torch.Tensor,
        decoded_s_final: torch.Tensor,
        metadata: Dict[str, Any],
        num_refine_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 0. Prepare Labels
        if y is None:
            # Assume all are normal if no labels provided
            y = torch.zeros(x.size(0), device=self.device)
        else:
            y = y.to(self.device)

        # 1. Reconstruction Loss (Margin-based)
        # Calculate squared error per element: (Batch, ...) matching x shape
        per_element_error = (decoded_s_final - x) ** 2

        # Average over all dimensions except batch to get per-sample error
        # dims to reduce: range(1, x.ndim) e.g., [1] for (B, D), [1, 2] for (B, S, D)
        reduce_dims = list(range(1, x.ndim))
        recon_errors = torch.mean(per_element_error, dim=reduce_dims)

        # Normal Loss (y=0): Minimize Error
        normal_mask = y == 0
        if normal_mask.any():
            loss_normal = recon_errors[normal_mask].mean()
        else:
            loss_normal = torch.tensor(0.0, device=self.device)

        # Anomaly Loss (y=1): Maximize Error (up to Margin)
        # Hinge Loss: max(0, margin - error)
        # We want error > margin. If error is small, loss is high.
        anomaly_mask = y == 1
        if anomaly_mask.any():
            dist_anomaly = recon_errors[anomaly_mask]
            loss_anomaly = torch.relu(self.margin - dist_anomaly).mean()
        else:
            loss_anomaly = torch.tensor(0.0, device=self.device)

        recon_loss_combined = loss_normal + (self.anomaly_weight * loss_anomaly)

        # 2. Stability Loss
        batch_stability_loss = self._compute_stability_loss(metadata, len(x), num_refine_steps)

        # 3. Entropy Loss (Helper from BaseObjective)
        probs = metadata["probs"]
        entropy_loss = self._compute_entropy(probs)

        # 4. Load Balancing Loss (Helper from BaseObjective)
        balance_loss = self._compute_load_balance_loss(probs)

        # Get coefficients from Config
        recon_coeff = self.config.objective.recon_coeff
        stability_coeff = self.config.objective.stability_coeff
        entropy_coeff = self.config.objective.entropy_coeff
        balance_coeff = self.config.objective.balance_coeff

        # Total Loss
        total_loss = (
            (recon_coeff * recon_loss_combined)
            + (stability_coeff * batch_stability_loss)
            + (entropy_coeff * entropy_loss)
            + (balance_coeff * balance_loss)
        )

        loss_components = {
            "total_loss": total_loss.item(),
            "recon_loss_normal": loss_normal.item(),
            "recon_loss_anomaly": loss_anomaly.item(),
            "stability_loss": batch_stability_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "balance_loss": balance_loss.item(),
        }

        return total_loss, loss_components
