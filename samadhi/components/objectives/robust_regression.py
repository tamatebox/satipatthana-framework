from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from samadhi.components.objectives.base_objective import BaseObjective
from samadhi.configs.main import SamadhiConfig


class RobustRegressionObjective(BaseObjective):
    """
    Supervised regression objective using Huber Loss (or L1 Loss) for robustness against outliers.

    Loss = HuberLoss(decoded, y) + Stability + Entropy + Balance
    """

    def __init__(self, config: SamadhiConfig, device: Optional[str] = None):
        if isinstance(config, dict):
            config = SamadhiConfig.from_dict(config)
        super().__init__(config, device)
        # Huber Loss is robust to outliers. delta determines the threshold between MSE and L1.
        self.loss_fn = nn.HuberLoss(delta=self.config.objective.huber_delta)

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
        if y is None:
            raise ValueError("Target 'y' cannot be None for RobustRegressionObjective.")

        # 1. Robust Reconstruction Loss
        recon_loss = self.loss_fn(decoded_s_final, y)

        # 2. Stability Loss
        batch_stability_loss = self._compute_stability_loss(metadata, len(x), num_refine_steps)

        # 3. Entropy Loss
        probs = metadata["probs"]
        entropy_loss = self._compute_entropy(probs)

        # 4. Load Balancing Loss
        balance_loss = self._compute_load_balance_loss(probs)

        # Coefficients
        recon_coeff = self.config.objective.recon_coeff
        stability_coeff = self.config.objective.stability_coeff
        entropy_coeff = self.config.objective.entropy_coeff
        balance_coeff = self.config.objective.balance_coeff

        total_loss = (
            (recon_coeff * recon_loss)
            + (stability_coeff * batch_stability_loss)
            + (entropy_coeff * entropy_loss)
            + (balance_coeff * balance_loss)
        )

        loss_components = {
            "total_loss": total_loss.item(),
            "robust_loss": recon_loss.item(),
            "stability_loss": batch_stability_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "balance_loss": balance_loss.item(),
        }

        return total_loss, loss_components
