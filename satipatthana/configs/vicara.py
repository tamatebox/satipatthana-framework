from dataclasses import dataclass
from satipatthana.configs.base import BaseConfig
from satipatthana.configs.enums import VicaraType


@dataclass
class BaseVicaraConfig(BaseConfig):
    dim: int = 64
    type: VicaraType = VicaraType.STANDARD
    refine_steps: int = 5
    inertia: float = 0.7
    training_attention_mode: str = "soft"
    prediction_attention_mode: str = "hard"

    def validate(self):
        if not (0.0 <= self.inertia <= 1.0):
            raise ValueError(f"Inertia must be between 0 and 1, got {self.inertia}")


@dataclass
class StandardVicaraConfig(BaseVicaraConfig):
    type: VicaraType = VicaraType.STANDARD


@dataclass
class WeightedVicaraConfig(BaseVicaraConfig):
    type: VicaraType = VicaraType.WEIGHTED


@dataclass
class ProbeVicaraConfig(BaseVicaraConfig):
    type: VicaraType = VicaraType.PROBE_SPECIFIC
    n_probes: int = 10
