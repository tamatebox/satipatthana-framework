"""
Satipatthana Framework - Introspective Recursive Attention Architecture.

Quick Start:
    from satipatthana import SatipatthanaConfig, create_system

    # Simple: create from preset
    system = create_system("mlp", input_dim=128, output_dim=10)

    # Custom: use config
    config = SatipatthanaConfig(
        input_dim=128,
        output_dim=10,
        latent_dim=64,
        adapter="mlp",
    )
    system = config.build()

Training:
    from satipatthana import CurriculumConfig, StageConfig
    from satipatthana.train import SatipatthanaTrainer

    curriculum = CurriculumConfig(
        stage1=StageConfig(epochs=20, learning_rate=1e-4),
    )
    trainer.run_curriculum(curriculum)
"""

from satipatthana.configs import (
    SatipatthanaConfig,
    CurriculumConfig,
    StageConfig,
    NoisePathRatios,
    create_system,
)
from satipatthana.core.system import SatipatthanaSystem, TrainingStage

__all__ = [
    # User-facing configs
    "SatipatthanaConfig",
    "CurriculumConfig",
    "StageConfig",
    "NoisePathRatios",
    # Factory
    "create_system",
    # Core
    "SatipatthanaSystem",
    "TrainingStage",
]

__version__ = "4.0.0"
