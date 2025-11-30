from dataclasses import dataclass
from src.configs.base import BaseConfig
from src.configs.enums import DecoderType


@dataclass(kw_only=True)
class BaseDecoderConfig(BaseConfig):
    dim: int = 64
    type: DecoderType = DecoderType.RECONSTRUCTION


@dataclass(kw_only=True)
class ReconstructionDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.RECONSTRUCTION
    input_dim: int  # Target dimension (Mandatory)
    decoder_hidden_dim: int = 64


@dataclass(kw_only=True)
class CnnDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.CNN
    channels: int  # Mandatory
    img_size: int  # Mandatory
    decoder_hidden_dim: int = 64


@dataclass(kw_only=True)
class LstmDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.LSTM
    output_dim: int  # Mandatory
    seq_len: int  # Mandatory
    decoder_hidden_dim: int = 128
    lstm_layers: int = 1


@dataclass(kw_only=True)
class SimpleSequenceDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.SIMPLE_SEQUENCE
    output_dim: int  # Mandatory
    seq_len: int  # Mandatory
    decoder_hidden_dim: int = 128
