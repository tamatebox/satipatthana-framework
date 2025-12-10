"""
FixedStepSati: Always runs for the maximum number of steps without early stopping.
"""

from typing import Tuple, Dict, Any
import torch

from samadhi.components.sati.base import BaseSati
from samadhi.core.santana import SantanaLog
from samadhi.configs.sati import FixedStepSatiConfig


class FixedStepSati(BaseSati):
    """
    Fixed Step Sati - never stops early.

    Always returns should_stop=False, allowing the loop to run
    for the full number of configured steps. Used when consistent
    step count is desired or for baseline comparisons.
    """

    def __init__(self, config: FixedStepSatiConfig = None):
        if config is None:
            config = FixedStepSatiConfig()
        super().__init__(config)

    def forward(
        self, current_state: torch.Tensor, santana: SantanaLog
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Never signal early stop.

        Args:
            current_state: Current state tensor (Batch, Dim)
            santana: SantanaLog containing state trajectory history

        Returns:
            should_stop: Always False
            info: Dictionary with step count and reason
        """
        step_count = len(santana)
        final_energy = santana.get_final_energy()

        return False, {
            "reason": "fixed_step_no_early_stop",
            "step_count": step_count,
            "energy": final_energy,
        }
