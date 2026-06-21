"""Mars weather fine-tuning helpers for Aurora."""

from mars_weather._paths import ensure_aurora_on_path

ensure_aurora_on_path()

from mars_weather.model import make_mars_aurora
from mars_weather.openmars import (
    ATMOS_VARS,
    MARS_STATIC_VARS,
    SURF_VARS,
    OpenMARSDataset,
    OpenMARSRolloutDataset,
    OpenMARSStats,
    collate_batch_pairs,
    collate_rollout_samples,
    compute_openmars_stats,
    load_openmars_stats,
    register_openmars_stats,
    save_openmars_stats,
)
from mars_weather.splits import (
    create_openmars_split_manifest,
    dataset_from_manifest,
    load_split_manifest,
    rollout_dataset_from_manifest,
    save_split_manifest,
)
from mars_weather.training import make_optimizer, normalized_mse_loss, normalized_mse_losses

__all__ = [
    "ATMOS_VARS",
    "MARS_STATIC_VARS",
    "SURF_VARS",
    "OpenMARSDataset",
    "OpenMARSRolloutDataset",
    "OpenMARSStats",
    "collate_batch_pairs",
    "collate_rollout_samples",
    "compute_openmars_stats",
    "create_openmars_split_manifest",
    "dataset_from_manifest",
    "load_openmars_stats",
    "load_split_manifest",
    "rollout_dataset_from_manifest",
    "make_mars_aurora",
    "make_optimizer",
    "normalized_mse_loss",
    "normalized_mse_losses",
    "register_openmars_stats",
    "save_split_manifest",
    "save_openmars_stats",
]
