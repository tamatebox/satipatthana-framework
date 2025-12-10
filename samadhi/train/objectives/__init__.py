"""
DEPRECATED: Objectives have been moved to samadhi/components/objectives/

This module provides backwards compatibility imports.
Please update your imports to use samadhi.components.objectives instead.
"""

# Re-export from new location for backwards compatibility
from samadhi.components.objectives import (
    BaseObjective,
    UnsupervisedObjective,
    AutoencoderObjective,
    AnomalyObjective,
    SupervisedClassificationObjective,
    SupervisedRegressionObjective,
    RobustRegressionObjective,
    CosineSimilarityObjective,
)

__all__ = [
    "BaseObjective",
    "UnsupervisedObjective",
    "AutoencoderObjective",
    "AnomalyObjective",
    "SupervisedClassificationObjective",
    "SupervisedRegressionObjective",
    "RobustRegressionObjective",
    "CosineSimilarityObjective",
]
