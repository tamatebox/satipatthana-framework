import pytest
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.adapters import MlpAdapterConfig, CnnAdapterConfig
from samadhi.configs.vicara import StandardVicaraConfig, ProbeVicaraConfig
from samadhi.configs.decoders import ReconstructionDecoderConfig


def test_samadhi_config_defaults():
    config = SamadhiConfig()
    assert config.dim == 64
    assert isinstance(config.adapter, MlpAdapterConfig)
    assert isinstance(config.vicara, StandardVicaraConfig)


def test_samadhi_config_nested_dict():
    data = {
        "dim": 128,
        "adapter": {"type": "cnn", "img_size": 32, "channels": 3},
        "vicara": {"type": "probe_specific", "n_probes": 5},
        "decoder": {"input_dim": 128},
    }
    config = SamadhiConfig.from_dict(data)

    assert config.dim == 128

    # Check adapter
    assert isinstance(config.adapter, CnnAdapterConfig)
    assert config.adapter.img_size == 32
    assert config.adapter.dim == 128  # Propagated

    # Check vicara
    assert isinstance(config.vicara, ProbeVicaraConfig)
    assert config.vicara.n_probes == 5
    assert config.vicara.dim == 128  # Propagated


def test_samadhi_config_legacy_flat():
    # Simulate legacy flat config dictionary
    data = {
        "dim": 256,
        "input_dim": 20,  # For MLP Adapter
        "type": "mlp",  # Adapter Type
        "gate_threshold": 0.8,  # For Vitakka
    }

    config = SamadhiConfig.from_dict(data)
    assert config.dim == 256

    # Adapter should have picked up input_dim
    assert isinstance(config.adapter, MlpAdapterConfig)
    assert config.adapter.input_dim == 20

    # Vitakka should have picked up gate_threshold
    assert config.vitakka.gate_threshold == 0.8
