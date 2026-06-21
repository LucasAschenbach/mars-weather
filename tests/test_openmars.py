from __future__ import annotations

from pathlib import Path

import torch

from mars_weather.model import make_mars_aurora
from mars_weather.openmars import (
    ATMOS_VARS,
    MARS_STATIC_VARS,
    SURF_VARS,
    OpenMARSDataset,
    compute_openmars_stats,
    open_openmars,
    register_openmars_stats,
)
from mars_weather.splits import create_openmars_split_manifest, dataset_from_manifest


SAMPLE = Path("data/49962798.nc")


def test_openmars_coordinates_and_shapes():
    with open_openmars(SAMPLE) as ds:
        assert ds.sizes["time"] == 360
        assert ds.sizes["lat"] == 36
        assert ds.sizes["lon"] == 72
        assert ds.sizes["lev"] == 35
        assert bool((ds.lat.diff("lat") < 0).all())
        assert bool((ds.lon.diff("lon") > 0).all())
        assert float(ds.lon.min()) >= 0
        assert float(ds.lon.max()) < 360


def test_dataset_emits_aurora_batches():
    dataset = OpenMARSDataset([SAMPLE])
    batch, target = dataset[0]

    assert len(dataset) == 358
    assert batch.metadata.atmos_levels == dataset.level_ids
    assert len(batch.metadata.atmos_levels) == 35
    assert set(batch.surf_vars) == set(SURF_VARS)
    assert set(batch.static_vars) == set(MARS_STATIC_VARS)
    assert set(batch.atmos_vars) == set(ATMOS_VARS)

    for value in batch.surf_vars.values():
        assert value.shape == (1, 2, 36, 72)
    for value in target.surf_vars.values():
        assert value.shape == (1, 1, 36, 72)
    for value in batch.atmos_vars.values():
        assert value.shape == (1, 2, 35, 36, 72)
    for value in target.atmos_vars.values():
        assert value.shape == (1, 1, 35, 36, 72)


def test_openmars_stats_register_and_normalise():
    stats = compute_openmars_stats([SAMPLE], max_time_steps=2)
    register_openmars_stats(stats)
    dataset = OpenMARSDataset([SAMPLE])
    batch, _ = dataset[0]
    normalized = batch.normalise(surf_stats={})

    for value in normalized.surf_vars.values():
        assert torch.isfinite(value).all()
    for value in normalized.static_vars.values():
        assert torch.isfinite(value).all()
    for value in normalized.atmos_vars.values():
        assert torch.isfinite(value).all()


def test_mars_model_constructor_without_checkpoint():
    dataset = OpenMARSDataset([SAMPLE])
    model = make_mars_aurora(level_ids=dataset.level_ids, size="small", load_checkpoint=False)

    assert model.surf_vars == SURF_VARS
    assert model.atmos_vars == ATMOS_VARS
    assert model.encoder.level_condition == dataset.level_ids


def test_target_year_split_filters_by_target_frame():
    train = OpenMARSDataset([SAMPLE], target_mars_years=(28,))
    val = OpenMARSDataset([SAMPLE], target_mars_years=(29,))

    assert len(train) == 0
    assert len(val) == 358


def test_split_manifest_round_trip():
    manifest = create_openmars_split_manifest(
        [SAMPLE],
        train_years=(28,),
        val_years=(29,),
        root=Path("."),
    )
    train = dataset_from_manifest(manifest, "train")
    val = dataset_from_manifest(manifest, "val")

    assert manifest["format"] == "openmars-target-year-split-v1"
    assert manifest["splits"]["train"]["num_samples"] == 0
    assert manifest["splits"]["val"]["num_samples"] == 358
    assert len(train) == 0
    assert len(val) == 358
