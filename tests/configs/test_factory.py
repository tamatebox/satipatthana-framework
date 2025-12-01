import pytest
from samadhi.configs.enums import AdapterType, VicaraType, DecoderType
from samadhi.configs.factory import (
    create_adapter_config,
    create_vicara_config,
    create_decoder_config,
    create_vitakka_config,
)
from samadhi.configs.adapters import MlpAdapterConfig, CnnAdapterConfig
from samadhi.configs.vicara import StandardVicaraConfig, ProbeVicaraConfig
from samadhi.configs.decoders import ReconstructionDecoderConfig, LstmDecoderConfig


def test_create_adapter_config_default():
    data = {"input_dim": 20}  # Missing type, should default to MLP
    config = create_adapter_config(data)
    assert isinstance(config, MlpAdapterConfig)
    assert config.input_dim == 20


def test_create_adapter_config_explicit():
    data = {"type": "cnn", "img_size": 64, "channels": 3}
    config = create_adapter_config(data)
    assert isinstance(config, CnnAdapterConfig)
    assert config.img_size == 64


def test_create_vicara_config_probe():
    data = {"type": "probe_specific", "n_probes": 5}
    config = create_vicara_config(data)
    assert isinstance(config, ProbeVicaraConfig)
    assert config.n_probes == 5


def test_create_decoder_config_fallback():
    # Test fallback: decoder_hidden_dim not set, adapter_hidden_dim present
    data = {"type": "reconstruction", "input_dim": 10, "adapter_hidden_dim": 128}
    config = create_decoder_config(data)
    assert isinstance(config, ReconstructionDecoderConfig)
    assert config.decoder_hidden_dim == 128


def test_create_lstm_decoder_config_fallback():
    # Test fallback: output_dim missing -> input_dim, decoder_hidden -> adapter_hidden
    data = {"type": "lstm", "input_dim": 30, "adapter_hidden_dim": 256, "seq_len": 10}
    config = create_decoder_config(data)
    assert isinstance(config, LstmDecoderConfig)
    assert config.output_dim == 30
    assert config.decoder_hidden_dim == 256
