"""
Curriculum Configuration for Satipatthana 4-Stage Training.

This module provides:
- StageConfig: Per-stage training settings
- NoisePathRatios: Stage 2 noise path distribution
- CurriculumConfig: Full curriculum configuration

Usage:
    # Simple: use defaults
    curriculum = CurriculumConfig()

    # Custom: adjust specific stages
    curriculum = CurriculumConfig(
        stage1=StageConfig(epochs=20, learning_rate=1e-4),
        stage2=StageConfig(epochs=10, noise_level=0.5),
    )

    # Pass to trainer
    trainer.run_curriculum(curriculum)
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class NoisePathRatios:
    """
    Stage 2 noise path distribution ratios.

    These ratios control how training batches are distributed across
    the four noise generation strategies in Stage 2:
    - clean: Original data (target=1.0)
    - augmented: Environmental noise via Augmenter (target=1.0-severity)
    - drunk: Internal dysfunction via drunk_mode (target=0.0)
    - mismatch: Logical inconsistency via batch shuffling (target=0.0)
    - void: Out-of-distribution data (target=0.0)

    Ratios should sum to 1.0.
    """

    clean: float = 0.2
    augmented: float = 0.2
    drunk: float = 0.2
    mismatch: float = 0.2
    void: float = 0.2

    def __post_init__(self):
        """Validate ratios sum to 1.0."""
        total = self.clean + self.augmented + self.drunk + self.mismatch + self.void
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"NoisePathRatios must sum to 1.0, got {total}")

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "clean": self.clean,
            "augmented": self.augmented,
            "drunk": self.drunk,
            "mismatch": self.mismatch,
            "void": self.void,
        }


@dataclass
class StageConfig:
    """
    Per-stage training configuration.

    Minimal configuration focused on the most important per-stage settings.
    Additional parameters can be added as needed.

    Args:
        epochs: Number of training epochs for this stage
        learning_rate: Learning rate for this stage (None = use trainer default)
        batch_size: Batch size for this stage (None = use trainer default)
        noise_level: Noise intensity for augmentation (Stage 1, 2)
        freeze: List of component names to freeze (overrides default policy)
        unfreeze: List of component names to unfreeze (overrides default policy)

    Stage-specific weights (applied only when relevant):
        stability_weight: Weight for stability loss (Stage 1)
        guidance_weight: Weight for label guidance loss (Stage 1)
        recon_weight: Weight for reconstruction loss (Stage 0, 1)
        diversity_weight: Weight for probe diversity loss (Stage 1)
    """

    epochs: int = 5
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None

    # Noise settings
    noise_level: Optional[float] = None

    # Freeze/unfreeze overrides (empty = use default stage policy)
    freeze: Optional[List[str]] = None
    unfreeze: Optional[List[str]] = None

    # Loss weights (None = use trainer default)
    stability_weight: Optional[float] = None
    guidance_weight: Optional[float] = None
    recon_weight: Optional[float] = None
    diversity_weight: Optional[float] = None


@dataclass
class CurriculumConfig:
    """
    Full 4-stage curriculum configuration.

    Provides stage-specific settings for each training phase:
    - Stage 0: Adapter Pre-training (Reconstruction)
    - Stage 1: Samatha Training (Stability + optional Label Guidance)
    - Stage 2: Vipassana Training (Contrastive Learning)
    - Stage 3: Decoder Fine-tuning (Task-specific)

    Args:
        stage0: Configuration for Adapter Pre-training
        stage1: Configuration for Samatha Training
        stage2: Configuration for Vipassana Training
        stage3: Configuration for Decoder Fine-tuning
        noise_path_ratios: Stage 2 noise path distribution

    Example:
        # Default curriculum
        curriculum = CurriculumConfig()

        # Custom curriculum with adjusted learning rates
        curriculum = CurriculumConfig(
            stage0=StageConfig(epochs=5, learning_rate=1e-3),
            stage1=StageConfig(epochs=10, learning_rate=5e-4, stability_weight=0.2),
            stage2=StageConfig(epochs=5, learning_rate=1e-4),
            stage3=StageConfig(epochs=5, learning_rate=1e-4),
        )

        # Skip Stage 0
        curriculum = CurriculumConfig(
            stage0=StageConfig(epochs=0),
        )
    """

    stage0: StageConfig = field(
        default_factory=lambda: StageConfig(
            epochs=5,
            learning_rate=1e-3,
            recon_weight=1.0,
        )
    )

    stage1: StageConfig = field(
        default_factory=lambda: StageConfig(
            epochs=10,
            learning_rate=5e-4,
            noise_level=0.3,
            stability_weight=0.1,
            guidance_weight=1.0,
            recon_weight=1.0,
            diversity_weight=0.1,
        )
    )

    stage2: StageConfig = field(
        default_factory=lambda: StageConfig(
            epochs=5,
            learning_rate=1e-4,
            noise_level=0.3,
        )
    )

    stage3: StageConfig = field(
        default_factory=lambda: StageConfig(
            epochs=5,
            learning_rate=1e-4,
        )
    )

    noise_path_ratios: NoisePathRatios = field(default_factory=NoisePathRatios)

    def get_stage_config(self, stage_num: int) -> StageConfig:
        """Get configuration for a specific stage number (0-3)."""
        stages = [self.stage0, self.stage1, self.stage2, self.stage3]
        if 0 <= stage_num <= 3:
            return stages[stage_num]
        raise ValueError(f"Invalid stage number: {stage_num}. Must be 0-3.")

    def total_epochs(self) -> int:
        """Calculate total epochs across all stages."""
        return self.stage0.epochs + self.stage1.epochs + self.stage2.epochs + self.stage3.epochs


__all__ = [
    "StageConfig",
    "NoisePathRatios",
    "CurriculumConfig",
]
