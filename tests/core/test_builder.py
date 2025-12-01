import pytest
import torch
import torch.nn as nn
from dataclasses import asdict
from samadhi.core.builder import SamadhiBuilder
from samadhi.core.engine import SamadhiEngine
from samadhi.components.adapters.base import BaseAdapter
from samadhi.components.adapters.mlp import MlpAdapter
from samadhi.components.adapters.vision import CnnAdapter
from samadhi.components.decoders.base import BaseDecoder
from samadhi.components.decoders.reconstruction import ReconstructionDecoder
from samadhi.components.decoders.vision import CnnDecoder
from samadhi.components.vitakka.base import BaseVitakka
from samadhi.components.vitakka.standard import StandardVitakka
from samadhi.components.vicara.base import BaseVicara
from samadhi.components.vicara.standard import StandardVicara
from samadhi.components.vicara.weighted import WeightedVicara
from samadhi.components.vicara.probe_specific import ProbeVicara
from samadhi.components.refiners.mlp import MlpRefiner
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.adapters import MlpAdapterConfig, CnnAdapterConfig, BaseAdapterConfig
from samadhi.configs.vicara import StandardVicaraConfig, ProbeVicaraConfig, WeightedVicaraConfig, BaseVicaraConfig
from samadhi.configs.vitakka import (
    StandardVitakkaConfig as VitakkaStandardConfig,
    BaseVitakkaConfig,
)  # Alias to avoid conflict
from samadhi.configs.decoders import ReconstructionDecoderConfig, CnnDecoderConfig, BaseDecoderConfig
from samadhi.configs.enums import AdapterType, VicaraType, DecoderType


# Mock implementations for testing the builder
class MockAdapter(BaseAdapter):

    def __init__(self, config: BaseAdapterConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_dim, config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockDecoder(BaseDecoder):
    def __init__(self, config: BaseDecoderConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.dim, config.input_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


class CustomVitakka(StandardVitakka):
    def __init__(self, config: BaseVitakkaConfig):
        super().__init__(config)


class CustomVicara(StandardVicara):
    def __init__(self, config: BaseVicaraConfig, refiner: MlpRefiner):
        super().__init__(config, refiner)


@pytest.fixture
def builder_config() -> SamadhiConfig:
    config_data = {
        "dim": 64,
        "seed": 42,
        "labels": ["A", "B", "C"],
        "objective": {
            "stability_coeff": 0.05,
            "entropy_coeff": 0.2,
            "balance_coeff": 0.005,
        },
        "adapter": {"type": AdapterType.MLP.value, "input_dim": 128, "dropout": 0.1},
        "vitakka": {"n_probes": 7, "gate_threshold": -0.5, "probe_trainable": True},
        "vicara": {"type": VicaraType.STANDARD.value, "refine_steps": 5, "inertia": 0.7},
        "decoder": {"type": DecoderType.RECONSTRUCTION.value, "input_dim": 10},
    }
    return SamadhiConfig.from_dict(config_data)


@pytest.fixture
def cnn_builder_config() -> SamadhiConfig:
    config_data = {
        "dim": 64,
        "adapter": {"type": AdapterType.CNN.value, "channels": 3, "img_size": 32},
        "decoder": {"type": DecoderType.CNN.value, "channels": 3, "img_size": 32, "input_dim": 32 * 32 * 3},
    }
    return SamadhiConfig.from_dict(config_data)


def test_builder_initialization(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    assert builder.config == builder_config
    assert builder.adapter is None
    assert builder.vitakka is None
    assert builder.vicara is None
    assert builder.decoder is None


def test_builder_set_adapter(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    adapter = MockAdapter(builder_config.adapter)
    builder.set_adapter(adapter)
    assert builder.adapter == adapter


def test_builder_set_adapter_by_type_mlp(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    builder.set_adapter(type=AdapterType.MLP.value)
    assert isinstance(builder.adapter, MlpAdapter)
    assert builder.adapter.config.type == AdapterType.MLP


def test_builder_set_adapter_by_type_cnn(cnn_builder_config: SamadhiConfig):
    builder = SamadhiBuilder(cnn_builder_config)
    builder.set_adapter(type=AdapterType.CNN.value)
    assert isinstance(builder.adapter, CnnAdapter)
    assert builder.adapter.config.type == AdapterType.CNN


def test_builder_set_vitakka(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)

    # Test default Vitakka creation
    builder.set_vitakka()
    assert isinstance(builder.vitakka, StandardVitakka)
    assert builder.vitakka.config.n_probes == builder_config.vitakka.n_probes

    # Test custom Vitakka setting
    custom_vitakka = CustomVitakka(builder_config.vitakka)
    builder.set_vitakka(custom_vitakka)
    assert builder.vitakka == custom_vitakka


def test_builder_set_vicara(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)

    # Test default StandardVicara creation
    builder.set_vicara(refiner_type="mlp")
    assert isinstance(builder.vicara, StandardVicara)
    assert builder.vicara.config.type == VicaraType.STANDARD

    # Test ProbeVicara creation via type string
    probe_vicara_config_data = {
        "dim": builder_config.dim,
        "adapter": asdict(builder_config.adapter),
        "vitakka": asdict(builder_config.vitakka),
        "vicara": {"type": VicaraType.PROBE_SPECIFIC.value, "n_probes": 10},
        "decoder": asdict(builder_config.decoder),
    }
    probe_builder = SamadhiBuilder(SamadhiConfig.from_dict(probe_vicara_config_data))
    probe_builder.set_vicara(vicara_type=VicaraType.PROBE_SPECIFIC.value, refiner_type="mlp")
    assert isinstance(probe_builder.vicara, ProbeVicara)
    assert probe_builder.vicara.config.n_probes == 10

    # Test WeightedVicara creation via type string
    weighted_vicara_config_data = {
        "dim": builder_config.dim,
        "adapter": asdict(builder_config.adapter),
        "vitakka": asdict(builder_config.vitakka),
        "vicara": {"type": VicaraType.WEIGHTED.value},
        "decoder": asdict(builder_config.decoder),
    }
    weighted_builder = SamadhiBuilder(SamadhiConfig.from_dict(weighted_vicara_config_data))
    weighted_builder.set_vicara(vicara_type=VicaraType.WEIGHTED.value, refiner_type="mlp")
    assert isinstance(weighted_builder.vicara, WeightedVicara)
    assert weighted_builder.vicara.config.type == VicaraType.WEIGHTED

    # Test custom Vicara setting
    custom_vicara = CustomVicara(builder_config.vicara, MlpRefiner(builder_config.vicara))
    builder.set_vicara(custom_vicara)
    assert builder.vicara == custom_vicara


def test_builder_set_decoder(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    decoder = MockDecoder(builder_config.decoder)
    builder.set_decoder(decoder)
    assert builder.decoder == decoder


def test_builder_set_decoder_by_type_reconstruction(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    builder.set_decoder(type=DecoderType.RECONSTRUCTION.value)
    assert isinstance(builder.decoder, ReconstructionDecoder)
    assert builder.decoder.config.type == DecoderType.RECONSTRUCTION


def test_builder_set_decoder_by_type_cnn(cnn_builder_config: SamadhiConfig):
    builder = SamadhiBuilder(cnn_builder_config)
    builder.set_decoder(type=DecoderType.CNN.value)
    assert isinstance(builder.decoder, CnnDecoder)
    assert builder.decoder.config.type == DecoderType.CNN


def test_builder_build_success(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    adapter = MockAdapter(builder_config.adapter)
    decoder = MockDecoder(builder_config.decoder)

    engine = builder.set_adapter(adapter).set_vitakka().set_vicara(refiner_type="mlp").set_decoder(decoder).build()

    assert isinstance(engine, SamadhiEngine)
    assert engine.adapter == adapter
    assert isinstance(engine.vitakka, StandardVitakka)
    assert isinstance(engine.vicara, StandardVicara)
    assert engine.decoder == decoder
    assert engine.config == builder_config


def test_builder_build_failure_missing_components(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    # Only set adapter
    builder.set_adapter(MockAdapter(builder_config.adapter))

    with pytest.raises(
        ValueError, match=r"All components \(adapter, vitakka, vicara, decoder\) must be set before building\."
    ):
        builder.build()


def test_builder_build_with_custom_components(builder_config: SamadhiConfig):
    builder = SamadhiBuilder(builder_config)
    adapter = MockAdapter(builder_config.adapter)
    vitakka = CustomVitakka(builder_config.vitakka)
    vicara = CustomVicara(builder_config.vicara, MlpRefiner(builder_config.vicara))
    decoder = MockDecoder(builder_config.decoder)

    engine = builder.set_adapter(adapter).set_vitakka(vitakka).set_vicara(vicara).set_decoder(decoder).build()

    assert isinstance(engine, SamadhiEngine)
    assert engine.adapter == adapter
    assert isinstance(engine.vitakka, CustomVitakka)
    assert engine.vicara == vicara
    assert engine.decoder == decoder
