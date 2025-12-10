"""
ThresholdSati: Stops when energy falls below threshold.
"""

from typing import Tuple, Dict, Any
import torch

from samadhi.components.sati.base import BaseSati
from samadhi.core.santana import SantanaLog
from samadhi.configs.sati import ThresholdSatiConfig


class ThresholdSati(BaseSati):
    """
    Threshold-based Sati - stops when convergence energy is low enough.

    Monitors the energy (state change magnitude) and signals stop
    when it falls below the configured threshold. Respects minimum
    step count before allowing early stopping.
    """

    def __init__(self, config: ThresholdSatiConfig = None):
        if config is None:
            config = ThresholdSatiConfig()
        super().__init__(config)

    def forward(
        self, current_state: torch.Tensor, santana: SantanaLog
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if energy is below threshold for early stopping.

        Args:
            current_state: Current state tensor (Batch, Dim)
            santana: SantanaLog containing state trajectory history

        Returns:
            should_stop: True if energy < threshold and min_steps reached
            info: Dictionary with convergence details
        """
        step_count = len(santana)
        final_energy = santana.get_final_energy()

        # Don't stop before min_steps
        if step_count < self.config.min_steps:
            return False, {
                "reason": "min_steps_not_reached",
                "step_count": step_count,
                "min_steps": self.config.min_steps,
                "energy": final_energy,
                "threshold": self.config.energy_threshold,
            }

        # Check if energy is available
        if final_energy is None:
            return False, {
                "reason": "no_energy_recorded",
                "step_count": step_count,
                "min_steps": self.config.min_steps,
                "energy": None,
                "threshold": self.config.energy_threshold,
            }

        # Check threshold condition
        should_stop = final_energy < self.config.energy_threshold

        if should_stop:
            reason = "threshold_reached"
        else:
            reason = "above_threshold"

        return should_stop, {
            "reason": reason,
            "step_count": step_count,
            "min_steps": self.config.min_steps,
            "energy": final_energy,
            "threshold": self.config.energy_threshold,
        }
