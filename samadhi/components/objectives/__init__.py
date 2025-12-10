"""
Objectives: Training objective components for Samadhi Framework v4.0.

Objectives define the loss functions used during training.
Moved from samadhi/train/objectives/ for better organization.
"""

from samadhi.components.objectives.base_objective import BaseObjective
from samadhi.components.objectives.unsupervised import UnsupervisedObjective
from samadhi.components.objectives.autoencoder import AutoencoderObjective
from samadhi.components.objectives.anomaly import AnomalyObjective
from samadhi.components.objectives.supervised_classification import (
    SupervisedClassificationObjective,
)
from samadhi.components.objectives.supervised_regression import (
    SupervisedRegressionObjective,
)
from samadhi.components.objectives.robust_regression import RobustRegressionObjective
from samadhi.components.objectives.cosine_similarity import CosineSimilarityObjective

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
