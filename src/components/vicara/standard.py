from typing import Dict, Any
import torch
import torch.nn as nn

from src.components.vicara.base import BaseVicara
from src.components.refiners.base import BaseRefiner


class StandardVicara(BaseVicara):
    """
    Standard Vicāra.
    Performs purification using a single refiner network.
    """

    def __init__(self, config: Dict[str, Any], refiner: BaseRefiner):
        super().__init__(config, nn.ModuleList([refiner]))
        # StandardVicara expects a single refiner, passed as a list to the base class

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        # Use the single shared refiner
        return self.refiners[0](s_t)
