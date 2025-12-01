from typing import Dict, Any
import torch
import torch.nn as nn

from samadhi.components.vicara.base import BaseVicara
from samadhi.components.refiners.base import BaseRefiner
from samadhi.configs.vicara import ProbeVicaraConfig
from samadhi.configs.factory import create_vicara_config


class ProbeVicara(BaseVicara):
    """
    Probe-Specific Vicāra.
    Possesses a distinct refiner (purification logic) for each probe (concept).
    """

    def __init__(self, config: ProbeVicaraConfig, refiners: nn.ModuleList[BaseRefiner]):
        if isinstance(config, dict):
            config = create_vicara_config(config)
        # ProbeVicara expects a list of refiners, one for each probe
        super().__init__(config, refiners)
        self.n_probes = self.config.n_probes
        if len(refiners) != self.n_probes:
            raise ValueError(f"ProbeVicara expects {self.n_probes} refiners, but got {len(refiners)}")

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        # Determine mode based on training status
        if self.training:
            mode = self.config.training_attention_mode
        else:
            mode = self.config.prediction_attention_mode

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
