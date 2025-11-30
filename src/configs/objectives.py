from dataclasses import dataclass
from src.configs.base import BaseConfig


@dataclass
class ObjectiveConfig(BaseConfig):
    """
    Configuration for training objectives (loss functions).
    """

    stability_coeff: float = 0.01
    entropy_coeff: float = 0.1
    balance_coeff: float = 0.001
    recon_coeff: float = 1.0
    huber_delta: float = 1.0
    anomaly_margin: float = 5.0
    anomaly_weight: float = 1.0
