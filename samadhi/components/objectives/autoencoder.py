from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from samadhi.components.objectives.base_objective import BaseObjective
from samadhi.configs.main import SamadhiConfig


class AutoencoderObjective(BaseObjective):
    """
    Autoencoder objective function. Computes only reconstruction loss.
    Vitakka and Vicara processes are skipped.
    """

    needs_vitakka: bool = False
    needs_vicara: bool = False

    def __init__(self, config: SamadhiConfig, device: Optional[str] = None):
        if isinstance(config, dict):
            config = SamadhiConfig.from_dict(config)
        super().__init__(config, device)
        self.recon_loss_fn = nn.MSELoss()

    def compute_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        s0: torch.Tensor,  # If Vitakka is skipped, this will be the Adapter output.
        s_final: torch.Tensor,  # If Vicara is skipped, this will be the same as s0.
        decoded_s_final: torch.Tensor,
        metadata: Dict[str, Any],
        num_refine_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # In autoencoder mode, s_final is adapter output (i.e., s0),
        # and decoded_s_final is the result of decoding it. The target is the original input x.
        recon_loss = self.recon_loss_fn(decoded_s_final, x)

        recon_coeff = self.config.objective.recon_coeff
        total_loss = recon_coeff * recon_loss

        loss_components = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "stability_loss": 0.0,  # Skipped, so 0
            "entropy_loss": 0.0,  # Skipped, so 0
            "balance_loss": 0.0,  # Skipped, so 0
        }

        return total_loss, loss_components
