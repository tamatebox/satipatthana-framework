import pytest
import torch
from samadhi.presets.tabular import create_mlp_samadhi
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.enums import AdapterType, VicaraType, DecoderType


@pytest.fixture
def tabular_config() -> SamadhiConfig:
    config_data = {
        "dim": 32,
        "seed": 42,
        "adapter": {"type": AdapterType.MLP.value, "input_dim": 64},
        "vitakka": {"n_probes": 5, "gate_threshold": -1.0},
        "vicara": {"type": VicaraType.STANDARD.value, "refine_steps": 2},
        "decoder": {"type": DecoderType.RECONSTRUCTION.value, "input_dim": 64},
    }
    return SamadhiConfig.from_dict(config_data)


def test_create_mlp_samadhi(tabular_config: SamadhiConfig):
    model = create_mlp_samadhi(tabular_config)
    assert model is not None

    # Check forward pass
    x = torch.randn(2, tabular_config.adapter.input_dim)
    output, s_final, meta = model(x)

    assert output.shape == (2, tabular_config.decoder.input_dim)
    assert s_final.shape == (2, tabular_config.dim)
