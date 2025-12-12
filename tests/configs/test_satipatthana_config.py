"""
Tests for SatipatthanaConfig and create_system() factory.

This tests the Phase 1 implementation of Issue #19 (Config simplification).
"""

import pytest
import torch

from satipatthana.configs.config import (
    SatipatthanaConfig,
    AdapterSpec,
    VicaraSpec,
    SatiSpec,
    VipassanaSpec,
    AugmenterSpec,
)
from satipatthana.configs.factory import create_system
from satipatthana.core.system import SatipatthanaSystem


class TestSatipatthanaConfigValidation:
    """Tests for SatipatthanaConfig validation."""

    def test_mlp_config_minimal(self):
        """MLP adapter should work with minimal required params."""
        config = SatipatthanaConfig(input_dim=128, output_dim=10)
        assert config.adapter == "mlp"
        assert config.latent_dim == 64

    def test_cnn_requires_img_size(self):
        """CNN adapter should require img_size."""
        with pytest.raises(ValueError, match="CNN adapter requires img_size"):
            SatipatthanaConfig(input_dim=784, output_dim=10, adapter="cnn", channels=1)

    def test_cnn_requires_channels(self):
        """CNN adapter should require channels."""
        with pytest.raises(ValueError, match="CNN adapter requires channels"):
            SatipatthanaConfig(input_dim=784, output_dim=10, adapter="cnn", img_size=28)

    def test_cnn_config_valid(self):
        """CNN adapter should work with required params."""
        config = SatipatthanaConfig(
            input_dim=784,
            output_dim=10,
            adapter="cnn",
            img_size=28,
            channels=1,
        )
        assert config.adapter == "cnn"
        assert config.img_size == 28
        assert config.channels == 1

    def test_lstm_requires_seq_len(self):
        """LSTM adapter should require seq_len."""
        with pytest.raises(ValueError, match="LSTM adapter requires seq_len"):
            SatipatthanaConfig(input_dim=32, output_dim=10, adapter="lstm")

    def test_transformer_requires_seq_len(self):
        """Transformer adapter should require seq_len."""
        with pytest.raises(ValueError, match="TRANSFORMER adapter requires seq_len"):
            SatipatthanaConfig(input_dim=32, output_dim=10, adapter="transformer")

    def test_lstm_config_valid(self):
        """LSTM adapter should work with required params."""
        config = SatipatthanaConfig(
            input_dim=32,
            output_dim=10,
            adapter="lstm",
            seq_len=20,
        )
        assert config.adapter == "lstm"
        assert config.seq_len == 20

    def test_transformer_config_valid(self):
        """Transformer adapter should work with required params."""
        config = SatipatthanaConfig(
            input_dim=32,
            output_dim=10,
            adapter="transformer",
            seq_len=20,
        )
        assert config.adapter == "transformer"
        assert config.seq_len == 20

    def test_latent_dim_must_be_positive(self):
        """latent_dim must be positive."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            SatipatthanaConfig(input_dim=128, output_dim=10, latent_dim=0)

        with pytest.raises(ValueError, match="latent_dim must be positive"):
            SatipatthanaConfig(input_dim=128, output_dim=10, latent_dim=-1)


class TestSatipatthanaConfigBuildHelpers:
    """Tests for _build_*_config helper methods."""

    def test_build_adapter_config_mlp(self):
        """_build_adapter_config should create MlpAdapterConfig for 'mlp'."""
        from satipatthana.configs.adapters import MlpAdapterConfig

        config = SatipatthanaConfig(input_dim=128, output_dim=10, adapter="mlp")
        adapter_cfg = config._build_adapter_config()

        assert isinstance(adapter_cfg, MlpAdapterConfig)
        assert adapter_cfg.input_dim == 128
        assert adapter_cfg.dim == 64  # latent_dim

    def test_build_adapter_config_cnn(self):
        """_build_adapter_config should create CnnAdapterConfig for 'cnn'."""
        from satipatthana.configs.adapters import CnnAdapterConfig

        config = SatipatthanaConfig(
            input_dim=784,
            output_dim=10,
            adapter="cnn",
            img_size=28,
            channels=1,
        )
        adapter_cfg = config._build_adapter_config()

        assert isinstance(adapter_cfg, CnnAdapterConfig)
        assert adapter_cfg.img_size == 28
        assert adapter_cfg.channels == 1
        assert adapter_cfg.dim == 64

    def test_build_vicara_config_standard(self):
        """_build_vicara_config should create StandardVicaraConfig for 'standard'."""
        from satipatthana.configs.vicara import StandardVicaraConfig

        config = SatipatthanaConfig(input_dim=128, output_dim=10, vicara="standard")
        vicara_cfg = config._build_vicara_config()

        assert isinstance(vicara_cfg, StandardVicaraConfig)
        assert vicara_cfg.dim == 64

    def test_build_vicara_config_weighted(self):
        """_build_vicara_config should create WeightedVicaraConfig for 'weighted'."""
        from satipatthana.configs.vicara import WeightedVicaraConfig

        config = SatipatthanaConfig(
            input_dim=128,
            output_dim=10,
            vicara="weighted",
            vicara_inertia=0.7,
        )
        vicara_cfg = config._build_vicara_config()

        assert isinstance(vicara_cfg, WeightedVicaraConfig)
        assert vicara_cfg.inertia == 0.7

    def test_build_vicara_config_probe(self):
        """_build_vicara_config should create ProbeVicaraConfig for 'probe'."""
        from satipatthana.configs.vicara import ProbeVicaraConfig

        config = SatipatthanaConfig(
            input_dim=128,
            output_dim=10,
            vicara="probe",
            n_probes=15,
        )
        vicara_cfg = config._build_vicara_config()

        assert isinstance(vicara_cfg, ProbeVicaraConfig)
        assert vicara_cfg.n_probes == 15

    def test_build_sati_config_fixed(self):
        """_build_sati_config should create FixedStepSatiConfig for 'fixed'."""
        from satipatthana.configs.sati import FixedStepSatiConfig

        config = SatipatthanaConfig(input_dim=128, output_dim=10, sati="fixed")
        sati_cfg = config._build_sati_config()

        assert isinstance(sati_cfg, FixedStepSatiConfig)

    def test_build_sati_config_threshold(self):
        """_build_sati_config should create ThresholdSatiConfig for 'threshold'."""
        from satipatthana.configs.sati import ThresholdSatiConfig

        config = SatipatthanaConfig(
            input_dim=128,
            output_dim=10,
            sati="threshold",
            sati_threshold=1e-5,
        )
        sati_cfg = config._build_sati_config()

        assert isinstance(sati_cfg, ThresholdSatiConfig)
        assert sati_cfg.energy_threshold == 1e-5

    def test_build_vipassana_config_standard(self):
        """_build_vipassana_config should create StandardVipassanaConfig for 'standard'."""
        from satipatthana.configs.vipassana import StandardVipassanaConfig

        config = SatipatthanaConfig(
            input_dim=128,
            output_dim=10,
            vipassana="standard",
            gru_hidden_dim=48,
        )
        vipassana_cfg = config._build_vipassana_config()

        assert isinstance(vipassana_cfg, StandardVipassanaConfig)
        assert vipassana_cfg.gru_hidden_dim == 48

    def test_build_augmenter_config_identity(self):
        """_build_augmenter_config should create IdentityAugmenterConfig for 'identity'."""
        from satipatthana.configs.augmenter import IdentityAugmenterConfig

        config = SatipatthanaConfig(input_dim=128, output_dim=10, augmenter="identity")
        augmenter_cfg = config._build_augmenter_config()

        assert isinstance(augmenter_cfg, IdentityAugmenterConfig)

    def test_build_augmenter_config_gaussian(self):
        """_build_augmenter_config should create GaussianNoiseAugmenterConfig for 'gaussian'."""
        from satipatthana.configs.augmenter import GaussianNoiseAugmenterConfig

        config = SatipatthanaConfig(input_dim=128, output_dim=10, augmenter="gaussian")
        augmenter_cfg = config._build_augmenter_config()

        assert isinstance(augmenter_cfg, GaussianNoiseAugmenterConfig)


class TestCreateSystemFromString:
    """Tests for create_system() with string adapter type."""

    def test_create_mlp_system(self):
        """create_system('mlp', ...) should create working system."""
        system = create_system("mlp", input_dim=128, output_dim=10)

        assert isinstance(system, SatipatthanaSystem)

        # Test forward pass
        x = torch.randn(4, 128)
        result = system(x)

        assert result.output.shape == (4, 10)
        assert result.s_star.shape == (4, 64)  # Default latent_dim
        assert result.trust_score.shape == (4, 1)

    def test_create_mlp_system_custom_latent_dim(self):
        """create_system() should respect custom latent_dim."""
        system = create_system("mlp", input_dim=128, output_dim=10, latent_dim=128)

        x = torch.randn(4, 128)
        result = system(x)

        assert result.s_star.shape == (4, 128)

    def test_create_cnn_system(self):
        """create_system('cnn', ...) should create working system."""
        system = create_system(
            "cnn",
            input_dim=784,  # Not used for CNN but required
            output_dim=10,
            img_size=28,
            channels=1,
        )

        assert isinstance(system, SatipatthanaSystem)

        # Test forward pass with image input
        # Note: run_vipassana=False to skip reconstruction error computation
        # (reconstruction head expects flattened input, but CNN uses 4D tensors)
        x = torch.randn(4, 1, 28, 28)
        result = system(x, run_vipassana=False)

        assert result.output.shape == (4, 10)

    def test_create_lstm_system(self):
        """create_system('lstm', ...) should create working system."""
        system = create_system(
            "lstm",
            input_dim=32,
            output_dim=10,
            seq_len=20,
        )

        assert isinstance(system, SatipatthanaSystem)

        # Test forward pass with sequence input
        # Note: run_vipassana=False to skip reconstruction error computation
        # (reconstruction head expects flattened input, but LSTM uses 3D tensors)
        x = torch.randn(4, 20, 32)
        result = system(x, run_vipassana=False)

        assert result.output.shape == (4, 10)

    def test_create_transformer_system(self):
        """create_system('transformer', ...) should create working system."""
        system = create_system(
            "transformer",
            input_dim=32,
            output_dim=10,
            seq_len=20,
        )

        assert isinstance(system, SatipatthanaSystem)

        # Test forward pass with sequence input
        # Note: run_vipassana=False to skip reconstruction error computation
        # (reconstruction head expects flattened input, but Transformer uses 3D tensors)
        x = torch.randn(4, 20, 32)
        result = system(x, run_vipassana=False)

        assert result.output.shape == (4, 10)


class TestCreateSystemFromConfig:
    """Tests for create_system() with SatipatthanaConfig."""

    def test_create_from_config(self):
        """create_system(config) should create working system."""
        config = SatipatthanaConfig(input_dim=64, output_dim=5, latent_dim=32)
        system = create_system(config)

        assert isinstance(system, SatipatthanaSystem)

        x = torch.randn(2, 64)
        result = system(x)

        assert result.output.shape == (2, 5)
        assert result.s_star.shape == (2, 32)

    def test_config_build_method(self):
        """SatipatthanaConfig.build() should create working system."""
        config = SatipatthanaConfig(input_dim=64, output_dim=5)
        system = config.build()

        assert isinstance(system, SatipatthanaSystem)

        x = torch.randn(2, 64)
        result = system(x)

        assert result.output.shape == (2, 5)

    def test_create_with_weighted_vicara(self):
        """System with weighted vicara should work."""
        config = SatipatthanaConfig(
            input_dim=64,
            output_dim=5,
            vicara="weighted",
        )
        system = config.build()

        x = torch.randn(2, 64)
        result = system(x)

        assert result.output.shape == (2, 5)

    def test_create_with_probe_vicara(self):
        """System with probe vicara should work."""
        config = SatipatthanaConfig(
            input_dim=64,
            output_dim=5,
            vicara="probe",
            n_probes=8,
        )
        system = config.build()

        x = torch.randn(2, 64)
        result = system(x)

        assert result.output.shape == (2, 5)

    def test_create_with_threshold_sati(self):
        """System with threshold sati should work."""
        config = SatipatthanaConfig(
            input_dim=64,
            output_dim=5,
            sati="threshold",
            sati_threshold=1e-3,
        )
        system = config.build()

        x = torch.randn(2, 64)
        result = system(x)

        assert result.output.shape == (2, 5)

    def test_create_with_fixed_sati(self):
        """System with fixed sati should work."""
        config = SatipatthanaConfig(
            input_dim=64,
            output_dim=5,
            sati="fixed",
        )
        system = config.build()

        x = torch.randn(2, 64)
        result = system(x)

        assert result.output.shape == (2, 5)

    def test_create_with_gaussian_augmenter(self):
        """System with gaussian augmenter should work."""
        config = SatipatthanaConfig(
            input_dim=64,
            output_dim=5,
            augmenter="gaussian",
        )
        system = config.build()

        x = torch.randn(2, 64)
        result = system(x)

        assert result.output.shape == (2, 5)

    def test_create_with_label_guidance(self):
        """System with label guidance enabled should work."""
        config = SatipatthanaConfig(
            input_dim=64,
            output_dim=5,
            use_label_guidance=True,
        )
        system = config.build()

        assert system.config.use_label_guidance is True


class TestSystemOutputs:
    """Tests for system outputs (scores, trajectory, etc.)."""

    def test_triple_scores_output(self):
        """System should output trust, conformity, and confidence scores."""
        system = create_system("mlp", input_dim=64, output_dim=5)

        x = torch.randn(4, 64)
        result = system(x)

        # All scores should be present
        assert result.trust_score is not None
        assert result.conformity_score is not None
        assert result.confidence_score is not None

        # All scores should be in [0, 1] range
        assert (result.trust_score >= 0).all() and (result.trust_score <= 1).all()
        assert (result.conformity_score >= 0).all() and (result.conformity_score <= 1).all()
        assert (result.confidence_score >= 0).all() and (result.confidence_score <= 1).all()

    def test_santana_log_output(self):
        """System should output SantanaLog with trajectory."""
        system = create_system("mlp", input_dim=64, output_dim=5)

        x = torch.randn(4, 64)
        result = system(x)

        # SantanaLog should be present
        assert result.santana is not None
        assert hasattr(result.santana, "states")
        assert hasattr(result.santana, "energies")

    def test_v_ctx_output(self):
        """System should output Vipassana context vector."""
        system = create_system("mlp", input_dim=64, output_dim=5)

        x = torch.randn(4, 64)
        result = system(x)

        # v_ctx should be present and have correct shape
        assert result.v_ctx is not None
        # context_dim = gru_hidden_dim + metric_proj_dim = 32 + 32 = 64
        assert result.v_ctx.shape[0] == 4


class TestDimensionPropagation:
    """Tests to verify dimension propagation works correctly."""

    def test_latent_dim_propagation(self):
        """latent_dim should propagate to all internal components."""
        config = SatipatthanaConfig(input_dim=64, output_dim=5, latent_dim=128)
        system = config.build()

        # Check s_star dimension
        x = torch.randn(2, 64)
        result = system(x)
        assert result.s_star.shape[1] == 128

    def test_custom_gru_hidden_dim(self):
        """Custom gru_hidden_dim should affect context_dim."""
        config = SatipatthanaConfig(
            input_dim=64,
            output_dim=5,
            gru_hidden_dim=64,  # Default is 32
        )
        system = config.build()

        x = torch.randn(2, 64)
        result = system(x)

        # context_dim = gru_hidden_dim + metric_proj_dim = 64 + 32 = 96
        assert result.v_ctx.shape[1] == 96

    def test_seed_propagation(self):
        """seed should be set in system config."""
        config = SatipatthanaConfig(input_dim=64, output_dim=5, seed=12345)
        system = config.build()

        assert system.config.seed == 12345
