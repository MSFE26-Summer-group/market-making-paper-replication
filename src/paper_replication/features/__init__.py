"""Feature engineering for the Attn-LOB pretraining task (Guo et al., 2023)."""

from paper_replication.features.config import FeatureConfig
from paper_replication.features.dataset import (
    AttnLOBDataset,
    build_feature_dataset,
    chronological_split,
)
from paper_replication.features.labels import (
    calibrate_alpha,
    compute_l_t,
    direction_labels,
)

__all__ = [
    "FeatureConfig",
    "AttnLOBDataset",
    "build_feature_dataset",
    "chronological_split",
    "calibrate_alpha",
    "compute_l_t",
    "direction_labels",
]
