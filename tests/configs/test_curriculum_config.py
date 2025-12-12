"""
Tests for CurriculumConfig, StageConfig, and NoisePathRatios.

This tests the Phase 2 implementation of Issue #19 (Curriculum configuration).
"""

import pytest

from satipatthana.configs.curriculum import (
    StageConfig,
    NoisePathRatios,
    CurriculumConfig,
)


class TestNoisePathRatios:
    """Tests for NoisePathRatios validation."""

    def test_default_ratios_sum_to_one(self):
        """Default ratios should sum to 1.0."""
        ratios = NoisePathRatios()
        total = ratios.clean + ratios.augmented + ratios.drunk + ratios.mismatch + ratios.void
        assert abs(total - 1.0) < 1e-6

    def test_custom_ratios_valid(self):
        """Custom ratios that sum to 1.0 should work."""
        ratios = NoisePathRatios(
            clean=0.5,
            augmented=0.3,
            drunk=0.1,
            mismatch=0.05,
            void=0.05,
        )
        assert ratios.clean == 0.5
        assert ratios.augmented == 0.3

    def test_invalid_ratios_raise_error(self):
        """Ratios not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            NoisePathRatios(
                clean=0.5,
                augmented=0.5,
                drunk=0.5,  # Now totals 1.5
                mismatch=0.0,
                void=0.0,
            )

    def test_to_dict(self):
        """to_dict should return correct dictionary."""
        ratios = NoisePathRatios(clean=0.4, augmented=0.3, drunk=0.2, mismatch=0.05, void=0.05)
        d = ratios.to_dict()

        assert d["clean"] == 0.4
        assert d["augmented"] == 0.3
        assert d["drunk"] == 0.2
        assert d["mismatch"] == 0.05
        assert d["void"] == 0.05


class TestStageConfig:
    """Tests for StageConfig."""

    def test_default_values(self):
        """Default StageConfig should have sensible defaults."""
        config = StageConfig()
        assert config.epochs == 5
        assert config.learning_rate is None
        assert config.batch_size is None
        assert config.noise_level is None
        assert config.freeze is None
        assert config.unfreeze is None

    def test_custom_values(self):
        """StageConfig should accept custom values."""
        config = StageConfig(
            epochs=20,
            learning_rate=1e-4,
            batch_size=64,
            noise_level=0.5,
            stability_weight=0.2,
        )
        assert config.epochs == 20
        assert config.learning_rate == 1e-4
        assert config.batch_size == 64
        assert config.noise_level == 0.5
        assert config.stability_weight == 0.2

    def test_freeze_unfreeze_lists(self):
        """StageConfig should accept freeze/unfreeze lists."""
        config = StageConfig(
            epochs=10,
            freeze=["adapter", "vitakka"],
            unfreeze=["vicara"],
        )
        assert config.freeze == ["adapter", "vitakka"]
        assert config.unfreeze == ["vicara"]


class TestCurriculumConfig:
    """Tests for CurriculumConfig."""

    def test_default_curriculum(self):
        """Default CurriculumConfig should have 4 stages with defaults."""
        curriculum = CurriculumConfig()

        # Stage 0: Adapter Pre-training
        assert curriculum.stage0.epochs == 5
        assert curriculum.stage0.learning_rate == 1e-3

        # Stage 1: Samatha Training
        assert curriculum.stage1.epochs == 10
        assert curriculum.stage1.learning_rate == 5e-4
        assert curriculum.stage1.stability_weight == 0.1

        # Stage 2: Vipassana Training
        assert curriculum.stage2.epochs == 5
        assert curriculum.stage2.learning_rate == 1e-4

        # Stage 3: Decoder Fine-tuning
        assert curriculum.stage3.epochs == 5
        assert curriculum.stage3.learning_rate == 1e-4

    def test_custom_stage_config(self):
        """CurriculumConfig should accept custom StageConfigs."""
        curriculum = CurriculumConfig(
            stage1=StageConfig(epochs=20, learning_rate=1e-4),
            stage2=StageConfig(epochs=15),
        )

        # Custom values
        assert curriculum.stage1.epochs == 20
        assert curriculum.stage1.learning_rate == 1e-4
        assert curriculum.stage2.epochs == 15

        # Unchanged stages keep defaults
        assert curriculum.stage0.epochs == 5
        assert curriculum.stage3.epochs == 5

    def test_skip_stage(self):
        """Setting epochs=0 should allow skipping a stage."""
        curriculum = CurriculumConfig(
            stage0=StageConfig(epochs=0),
        )
        assert curriculum.stage0.epochs == 0

    def test_get_stage_config(self):
        """get_stage_config should return correct stage."""
        curriculum = CurriculumConfig()

        assert curriculum.get_stage_config(0) is curriculum.stage0
        assert curriculum.get_stage_config(1) is curriculum.stage1
        assert curriculum.get_stage_config(2) is curriculum.stage2
        assert curriculum.get_stage_config(3) is curriculum.stage3

    def test_get_stage_config_invalid(self):
        """get_stage_config should raise for invalid stage number."""
        curriculum = CurriculumConfig()

        with pytest.raises(ValueError, match="Invalid stage number"):
            curriculum.get_stage_config(4)

        with pytest.raises(ValueError, match="Invalid stage number"):
            curriculum.get_stage_config(-1)

    def test_total_epochs(self):
        """total_epochs should sum all stage epochs."""
        curriculum = CurriculumConfig()
        # Default: 5 + 10 + 5 + 5 = 25
        assert curriculum.total_epochs() == 25

        curriculum = CurriculumConfig(
            stage0=StageConfig(epochs=0),
            stage1=StageConfig(epochs=20),
            stage2=StageConfig(epochs=10),
            stage3=StageConfig(epochs=10),
        )
        # 0 + 20 + 10 + 10 = 40
        assert curriculum.total_epochs() == 40

    def test_noise_path_ratios_default(self):
        """Default CurriculumConfig should have default NoisePathRatios."""
        curriculum = CurriculumConfig()
        ratios = curriculum.noise_path_ratios

        assert ratios.clean == 0.2
        assert ratios.augmented == 0.2
        assert ratios.drunk == 0.2
        assert ratios.mismatch == 0.2
        assert ratios.void == 0.2

    def test_custom_noise_path_ratios(self):
        """CurriculumConfig should accept custom NoisePathRatios."""
        curriculum = CurriculumConfig(
            noise_path_ratios=NoisePathRatios(
                clean=0.3,
                augmented=0.3,
                drunk=0.2,
                mismatch=0.1,
                void=0.1,
            )
        )

        assert curriculum.noise_path_ratios.clean == 0.3
        assert curriculum.noise_path_ratios.void == 0.1


class TestCurriculumConfigUseCases:
    """Test real-world usage patterns."""

    def test_quick_experiment_curriculum(self):
        """Quick experiment with fewer epochs."""
        curriculum = CurriculumConfig(
            stage0=StageConfig(epochs=1),
            stage1=StageConfig(epochs=2),
            stage2=StageConfig(epochs=1),
            stage3=StageConfig(epochs=1),
        )
        assert curriculum.total_epochs() == 5

    def test_focus_on_samatha_curriculum(self):
        """Curriculum focusing on Samatha training."""
        curriculum = CurriculumConfig(
            stage0=StageConfig(epochs=0),
            stage1=StageConfig(
                epochs=50,
                learning_rate=1e-4,
                stability_weight=0.5,
                noise_level=0.2,
            ),
            stage2=StageConfig(epochs=5),
            stage3=StageConfig(epochs=5),
        )

        assert curriculum.stage0.epochs == 0
        assert curriculum.stage1.epochs == 50
        assert curriculum.stage1.stability_weight == 0.5

    def test_high_diversity_void_curriculum(self):
        """Curriculum with more void/mismatch training."""
        curriculum = CurriculumConfig(
            noise_path_ratios=NoisePathRatios(
                clean=0.1,
                augmented=0.1,
                drunk=0.2,
                mismatch=0.3,
                void=0.3,
            )
        )

        assert curriculum.noise_path_ratios.void == 0.3
        assert curriculum.noise_path_ratios.mismatch == 0.3
