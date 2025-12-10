from dataclasses import dataclass
from samadhi.configs.base import BaseConfig
from samadhi.configs.enums import DecoderType


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


@dataclass(kw_only=True)
class ConditionalDecoderConfig(BaseDecoderConfig):
    """
    Configuration for Conditional Decoder.

    Takes concatenated input of latent state and Vipassana context vector.
    """

    type: DecoderType = DecoderType.CONDITIONAL
    context_dim: int = 32  # Vipassana context vector dimension
    output_dim: int = 10  # Task-specific output dimension
    decoder_hidden_dim: int = 128


@dataclass(kw_only=True)
class SimpleAuxHeadConfig(BaseDecoderConfig):
    """
    Configuration for Simple Auxiliary Head.

    Used for Stage 1 label guidance (direct task prediction from S*).
    """

    type: DecoderType = DecoderType.SIMPLE_AUX_HEAD
    output_dim: int = 10  # Task-specific output dimension
    decoder_hidden_dim: int = 64
