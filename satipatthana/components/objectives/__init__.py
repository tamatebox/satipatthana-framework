"""
Objectives: Training objective components for Satipatthana Framework v4.0.

Provides loss functions for the 4-stage curriculum training:
- Stage 2: VipassanaObjective (contrastive learning)
- Stage 1: GuidanceLoss (label guidance), StabilityLoss (trajectory smoothness)
- Stage 3: Task-specific losses (CrossEntropy, MSE, etc.) via PyTorch directly
"""

from satipatthana.components.objectives.vipassana import (
    VipassanaObjective,
    GuidanceLoss,
    StabilityLoss,
)

__all__ = [
    "VipassanaObjective",
    "GuidanceLoss",
    "StabilityLoss",
]
