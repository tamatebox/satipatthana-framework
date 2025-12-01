from dataclasses import dataclass
from samadhi.configs.base import BaseConfig


@dataclass
class BaseVitakkaConfig(BaseConfig):
    dim: int = 64
    n_probes: int = 10
    probe_trainable: bool = True
    training_attention_mode: str = "soft"
    prediction_attention_mode: str = "hard"


@dataclass
class StandardVitakkaConfig(BaseVitakkaConfig):
    gate_threshold: float = 0.6
    mix_alpha: float = 0.5
    softmax_temp: float = 0.2
