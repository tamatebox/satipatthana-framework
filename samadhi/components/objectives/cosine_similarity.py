from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from samadhi.components.objectives.base_objective import BaseObjective
from samadhi.configs.main import SamadhiConfig


class CosineSimilarityObjective(BaseObjective):
    """
    Unsupervised objective that maximizes Cosine Similarity between input and reconstruction.
    Suitable for learning semantic alignment/direction rather than exact magnitude.

    Loss = CosineEmbeddingLoss(x, decoded, target=1) + Stability + Entropy + Balance
    """

    def __init__(self, config: SamadhiConfig, device: Optional[str] = None):
        if isinstance(config, dict):
            config = SamadhiConfig.from_dict(config)
        super().__init__(config, device)
        self.loss_fn = nn.CosineEmbeddingLoss()

    def compute_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],  # Ignored
        s0: torch.Tensor,
        s_final: torch.Tensor,
        decoded_s_final: torch.Tensor,
        metadata: Dict[str, Any],
        num_refine_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Cosine Embedding Loss
        # We want x and decoded_s_final to be similar (target=1)
        # Input to CosineEmbeddingLoss requires flattened vectors if they are multidimensional images
        # But typically used for vectors. If x is (B, D), it works directly.
        # If x is (B, C, H, W), we should flatten.

        x_flat = x.view(x.size(0), -1)
        decoded_flat = decoded_s_final.view(decoded_s_final.size(0), -1)

        target = torch.ones(x.size(0), device=self.device)
        cosine_loss = self.loss_fn(decoded_flat, x_flat, target)

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
            (recon_coeff * cosine_loss)
            + (stability_coeff * batch_stability_loss)
            + (entropy_coeff * entropy_loss)
            + (balance_coeff * balance_loss)
        )

        loss_components = {
            "total_loss": total_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "stability_loss": batch_stability_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "balance_loss": balance_loss.item(),
        }

        return total_loss, loss_components
