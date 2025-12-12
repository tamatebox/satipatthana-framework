from dataclasses import dataclass
from satipatthana.configs.base import BaseConfig
from satipatthana.configs.enums import AdapterType


@dataclass(kw_only=True)
class BaseAdapterConfig(BaseConfig):
    dim: int = 64
    type: AdapterType = AdapterType.MLP
    dropout: float = 0.1

    def validate(self):
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"Dropout must be between 0 and 1, got {self.dropout}")


@dataclass(kw_only=True)
class IdentityAdapterConfig(BaseAdapterConfig):
    """Config for IdentityAdapter. Passes input through unchanged."""

    type: AdapterType = AdapterType.IDENTITY
    input_dim: int  # Mandatory

    def __post_init__(self):
        self.dim = self.input_dim  # dim equals input_dim for identity


@dataclass(kw_only=True)
class MlpAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.MLP
    input_dim: int  # Mandatory
    adapter_hidden_dim: int = 256


@dataclass(kw_only=True)
class CnnAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.CNN
    channels: int  # Mandatory
    img_size: int  # Mandatory

    def validate(self):
        super().validate()
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")


@dataclass(kw_only=True)
class LstmAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.LSTM
    input_dim: int  # Mandatory
    seq_len: int  # Mandatory
    adapter_hidden_dim: int = 128
    lstm_layers: int = 1


@dataclass(kw_only=True)
class TransformerAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.TRANSFORMER
    input_dim: int  # Mandatory
    seq_len: int  # Mandatory
    adapter_hidden_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
