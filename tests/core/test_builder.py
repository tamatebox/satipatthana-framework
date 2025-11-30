import pytest
import torch
import torch.nn as nn
from src.core.builder import SamadhiBuilder
from src.core.engine import SamadhiEngine
from src.components.adapters.base import BaseAdapter
from src.components.decoders.base import BaseDecoder
from src.components.vitakka.base import BaseVitakka
from src.components.vitakka.standard import StandardVitakka
from src.components.vicara.base import BaseVicara
from src.components.vicara.standard import StandardVicara
from src.components.vicara.weighted import WeightedVicara
from src.components.vicara.probe_specific import ProbeVicara
from src.components.refiners.mlp import MlpRefiner


# Mock implementations for testing the builder
class MockAdapter(BaseAdapter):

    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config["input_dim"], config["dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockDecoder(BaseDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config["dim"], config["output_dim"])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


class CustomVitakka(StandardVitakka):
    pass  # Just a subclass to ensure custom Vitakka can be set


class CustomVicara(StandardVicara):
    pass  # Just a subclass to ensure custom Vicara can be set


@pytest.fixture
def builder_config():
    return {
        "dim": 64,
        "input_dim": 128,
        "output_dim": 10,
        "n_probes": 7,
        "gate_threshold": -0.5,
        "refine_steps": 5,
        "vicara_type": "standard",
        "probe_trainable": True,
    }


def test_builder_initialization(builder_config):
    builder = SamadhiBuilder(builder_config)
    assert builder.config == builder_config
    assert builder.adapter is None
    assert builder.vitakka is None
    assert builder.vicara is None
    assert builder.decoder is None


def test_builder_set_adapter(builder_config):
    builder = SamadhiBuilder(builder_config)
    adapter = MockAdapter(builder_config)
    builder.set_adapter(adapter)
    assert builder.adapter == adapter


def test_builder_set_vitakka(builder_config):
    builder = SamadhiBuilder(builder_config)

    # Test default Vitakka creation
    builder.set_vitakka()  # No adapter argument needed
    assert isinstance(builder.vitakka, StandardVitakka)

    # Test custom Vitakka setting
    custom_vitakka = CustomVitakka(builder_config)  # No adapter argument needed
    builder.set_vitakka(custom_vitakka)
    assert builder.vitakka == custom_vitakka


def test_builder_set_vicara(builder_config):
    builder = SamadhiBuilder(builder_config)

    # Test default StandardVicara creation
    builder.set_vicara(refiner_type="mlp")  # Pass refiner_type
    assert isinstance(builder.vicara, StandardVicara)

    # Test ProbeVicara creation via type string
    builder.set_vicara(vicara_type="probe_specific", refiner_type="mlp", n_probes=builder_config["n_probes"])
    assert isinstance(builder.vicara, ProbeVicara)

    # Test WeightedVicara creation via config override
    builder_config["vicara_type"] = "weighted"
    builder_with_weighted_config = SamadhiBuilder(builder_config)
    builder_with_weighted_config.set_vicara(refiner_type="mlp")  # Pass refiner_type
    assert isinstance(builder_with_weighted_config.vicara, WeightedVicara)

    # Test custom Vicara setting
    custom_vicara = CustomVicara(builder_config, nn.ModuleList([MlpRefiner(builder_config)]))
    builder.set_vicara(custom_vicara)
    assert builder.vicara == custom_vicara


def test_builder_set_decoder(builder_config):
    builder = SamadhiBuilder(builder_config)
    decoder = MockDecoder(builder_config)
    builder.set_decoder(decoder)
    assert builder.decoder == decoder


def test_builder_build_success(builder_config):
    builder = SamadhiBuilder(builder_config)
    adapter = MockAdapter(builder_config)
    decoder = MockDecoder(builder_config)

    engine = builder.set_adapter(adapter).set_vitakka().set_vicara(refiner_type="mlp").set_decoder(decoder).build()

    assert isinstance(engine, SamadhiEngine)
    assert engine.adapter == adapter
    assert isinstance(engine.vitakka, StandardVitakka)
    assert isinstance(engine.vicara, StandardVicara)
    assert engine.decoder == decoder
    assert engine.config == builder_config


def test_builder_build_failure_missing_components(builder_config):
    builder = SamadhiBuilder(builder_config)
    # Only set adapter
    builder.set_adapter(MockAdapter(builder_config))

    with pytest.raises(
        ValueError, match=r"All components \(adapter, vitakka, vicara, decoder\) must be set before building\."
    ):
        builder.build()


def test_builder_build_with_custom_components(builder_config):
    builder = SamadhiBuilder(builder_config)
    adapter = MockAdapter(builder_config)
    vitakka = CustomVitakka(builder_config)  # No adapter argument needed
    vicara = CustomVicara(builder_config, nn.ModuleList([MlpRefiner(builder_config)]))
    decoder = MockDecoder(builder_config)

    engine = builder.set_adapter(adapter).set_vitakka(vitakka).set_vicara(vicara).set_decoder(decoder).build()

    assert isinstance(engine, SamadhiEngine)
    assert engine.adapter == adapter
    assert isinstance(engine.vitakka, CustomVitakka)
    assert engine.vicara == vicara
    assert engine.decoder == decoder
