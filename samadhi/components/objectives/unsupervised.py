from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from samadhi.components.objectives.base_objective import BaseObjective
from samadhi.configs.main import SamadhiConfig


class UnsupervisedObjective(BaseObjective):
    """
    Unsupervised objective for Samadhi Model, typically using MSE for reconstruction
    against the input itself. Incorporates reconstruction, stability, entropy, and
    load balancing losses.
    """

    def __init__(self, config: SamadhiConfig, device: Optional[str] = None):
        if isinstance(config, dict):
            config = SamadhiConfig.from_dict(config)
        super().__init__(config, device)
        self.recon_loss_fn = nn.MSELoss()

    def compute_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],  # y is ignored in unsupervised learning
        s0: torch.Tensor,
        s_final: torch.Tensor,
        decoded_s_final: torch.Tensor,
        metadata: Dict[str, Any],
        num_refine_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 1. Reconstruction Loss (x is the target)
        recon_loss = self.recon_loss_fn(decoded_s_final, x)

        # 2. Stability Loss
        batch_stability_loss = self._compute_stability_loss(metadata, len(x), num_refine_steps)

        # 3. Entropy Loss
        probs = metadata["probs"]
        entropy_loss = self._compute_entropy(probs)

        # 4. Load Balancing Loss
        balance_loss = self._compute_load_balance_loss(probs)

        # Get coefficients from Config
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
            "recon_loss": recon_loss.item(),
            "stability_loss": batch_stability_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "balance_loss": balance_loss.item(),
        }

        return total_loss, loss_components
