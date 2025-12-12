"""
Tests for v4.0 Configuration classes.

Phase 1 Tests:
- Config nested structure
- Config validation
- Type enum correctness
"""

import pytest
from satipatthana.configs.enums import AugmenterType, SatiType, VipassanaType
from satipatthana.configs.augmenter import (
    BaseAugmenterConfig,
    IdentityAugmenterConfig,
    GaussianNoiseAugmenterConfig,
)
from satipatthana.configs.sati import (
    BaseSatiConfig,
    FixedStepSatiConfig,
    ThresholdSatiConfig,
)
from satipatthana.configs.vipassana import (
    BaseVipassanaConfig,
    StandardVipassanaConfig,
    LSTMVipassanaConfig,
)
from satipatthana.configs.system import (
    SamathaConfig,
    VipassanaEngineConfig,
    SystemConfig,
)


class TestAugmenterConfig:
    """Tests for Augmenter configurations."""

    def test_identity_augmenter_default(self):
        """Test IdentityAugmenterConfig defaults."""
        config = IdentityAugmenterConfig()
        assert config.type == AugmenterType.IDENTITY

    def test_gaussian_noise_augmenter_default(self):
        """Test GaussianNoiseAugmenterConfig defaults."""
        config = GaussianNoiseAugmenterConfig()
        assert config.type == AugmenterType.GAUSSIAN_NOISE
        assert config.max_noise_std == 0.1

    def test_gaussian_noise_augmenter_custom(self):
        """Test GaussianNoiseAugmenterConfig with custom values."""
        config = GaussianNoiseAugmenterConfig(max_noise_std=0.5)
        assert config.max_noise_std == 0.5

    def test_gaussian_noise_augmenter_validation_error(self):
        """Test GaussianNoiseAugmenterConfig validation."""
        with pytest.raises(ValueError, match="max_noise_std must be non-negative"):
            GaussianNoiseAugmenterConfig(max_noise_std=-0.1)


class TestSatiConfig:
    """Tests for Sati configurations."""

    def test_fixed_step_sati_default(self):
        """Test FixedStepSatiConfig defaults."""
        config = FixedStepSatiConfig()
        assert config.type == SatiType.FIXED_STEP

    def test_threshold_sati_default(self):
        """Test ThresholdSatiConfig defaults."""
        config = ThresholdSatiConfig()
        assert config.type == SatiType.THRESHOLD
        assert config.energy_threshold == 1e-4
        assert config.min_steps == 1

    def test_threshold_sati_custom(self):
        """Test ThresholdSatiConfig with custom values."""
        config = ThresholdSatiConfig(energy_threshold=0.01, min_steps=5)
        assert config.energy_threshold == 0.01
        assert config.min_steps == 5

    def test_threshold_sati_validation_threshold(self):
        """Test ThresholdSatiConfig validation for threshold."""
        with pytest.raises(ValueError, match="energy_threshold must be non-negative"):
            ThresholdSatiConfig(energy_threshold=-0.001)

    def test_threshold_sati_validation_min_steps(self):
        """Test ThresholdSatiConfig validation for min_steps."""
        with pytest.raises(ValueError, match="min_steps must be at least 1"):
            ThresholdSatiConfig(min_steps=0)


class TestVipassanaConfig:
    """Tests for Vipassana configurations."""

    def test_standard_vipassana_default(self):
        """Test StandardVipassanaConfig defaults."""
        config = StandardVipassanaConfig()
        assert config.type == VipassanaType.STANDARD
        # context_dim = gru_hidden_dim (32) + metric_proj_dim (32) = 64
        assert config.context_dim == 64
        assert config.gru_hidden_dim == 32
        assert config.metric_proj_dim == 32
        assert config.latent_dim == 64
        assert config.max_steps == 10

    def test_lstm_vipassana_default(self):
        """Test LSTMVipassanaConfig defaults."""
        config = LSTMVipassanaConfig()
        assert config.type == VipassanaType.LSTM
        assert config.context_dim == 32
        assert config.hidden_dim == 64
        assert config.num_layers == 1
        assert config.bidirectional is False

    def test_lstm_vipassana_custom(self):
        """Test LSTMVipassanaConfig with custom values."""
        config = LSTMVipassanaConfig(context_dim=64, hidden_dim=128, num_layers=2, bidirectional=True)
        assert config.context_dim == 64
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.bidirectional is True


class TestSystemConfig:
    """Tests for System-level configurations."""

    def test_samatha_config_default(self):
        """Test SamathaConfig defaults."""
        config = SamathaConfig()
        assert config.dim == 64
        assert config.max_steps == 10
        assert isinstance(config.augmenter, IdentityAugmenterConfig)
        assert isinstance(config.sati, FixedStepSatiConfig)

    def test_vipassana_engine_config_default(self):
        """Test VipassanaEngineConfig defaults."""
        config = VipassanaEngineConfig()
        assert isinstance(config.vipassana, StandardVipassanaConfig)

    def test_system_config_default(self):
        """Test SystemConfig defaults."""
        config = SystemConfig()
        assert config.dim == 64
        assert config.seed == 42
        assert config.use_label_guidance is False
        assert isinstance(config.samatha, SamathaConfig)
        assert isinstance(config.vipassana, VipassanaEngineConfig)

    def test_system_config_nested_structure(self):
        """Test SystemConfig nested structure access."""
        config = SystemConfig()

        # Access nested components
        assert config.samatha.augmenter.type == AugmenterType.IDENTITY
        assert config.samatha.sati.type == SatiType.FIXED_STEP
        assert config.vipassana.vipassana.type == VipassanaType.STANDARD

    def test_system_config_custom_components(self):
        """Test SystemConfig with custom component configs."""
        config = SystemConfig(
            dim=128,
            use_label_guidance=True,
            samatha=SamathaConfig(
                dim=128,
                max_steps=20,
                augmenter=GaussianNoiseAugmenterConfig(max_noise_std=0.2),
                sati=ThresholdSatiConfig(energy_threshold=0.001),
            ),
            vipassana=VipassanaEngineConfig(vipassana=LSTMVipassanaConfig(hidden_dim=128, num_layers=2)),
        )

        assert config.dim == 128
        assert config.use_label_guidance is True
        assert config.samatha.max_steps == 20
        assert config.samatha.augmenter.max_noise_std == 0.2
        assert config.samatha.sati.energy_threshold == 0.001
        assert config.vipassana.vipassana.num_layers == 2


class TestEnumTypes:
    """Tests for enum type correctness."""

    def test_augmenter_types(self):
        """Test AugmenterType enum values."""
        assert AugmenterType.IDENTITY.value == "identity"
        assert AugmenterType.GAUSSIAN_NOISE.value == "gaussian_noise"

    def test_sati_types(self):
        """Test SatiType enum values."""
        assert SatiType.FIXED_STEP.value == "fixed_step"
        assert SatiType.THRESHOLD.value == "threshold"

    def test_vipassana_types(self):
        """Test VipassanaType enum values."""
        assert VipassanaType.STANDARD.value == "standard"
        assert VipassanaType.LSTM.value == "lstm"
