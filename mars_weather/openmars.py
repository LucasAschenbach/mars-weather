"""OpenMARS data adapters for Aurora."""

from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from mars_weather._paths import ensure_aurora_on_path

ensure_aurora_on_path()

from aurora import Batch, Metadata
from aurora.normalisation import level_to_str, locations, scales

SURF_VARS: tuple[str, ...] = ("ps", "tsurf", "co2ice", "dustcol")
ATMOS_VARS: tuple[str, ...] = ("u", "v", "temp")
MARS_STATIC_VARS: tuple[str, ...] = ("mars_static",)

MARTIAN_SOL_SECONDS = 88775.244
MARTIAN_HOUR_SECONDS = MARTIAN_SOL_SECONDS / 24
OPENMARS_STEP_HOURS = 2
OPENMARS_STEP = timedelta(seconds=OPENMARS_STEP_HOURS * MARTIAN_HOUR_SECONDS)
OPENMARS_TIME_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


@dataclasses.dataclass(frozen=True)
class OpenMARSStats:
    """Normalisation statistics for OpenMARS variables."""

    locations: dict[str, float]
    scales: dict[str, float]
    sigma_levels: tuple[float, ...]
    level_ids: tuple[int, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "locations": self.locations,
            "scales": self.scales,
            "sigma_levels": list(self.sigma_levels),
            "level_ids": list(self.level_ids),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "OpenMARSStats":
        return cls(
            locations={str(k): float(v) for k, v in data["locations"].items()},
            scales={str(k): float(v) for k, v in data["scales"].items()},
            sigma_levels=tuple(float(v) for v in data["sigma_levels"]),
            level_ids=tuple(int(v) for v in data["level_ids"]),
        )


def open_openmars(path: str | Path) -> xr.Dataset:
    """Open an OpenMARS NetCDF file and normalize coordinates for Aurora."""

    ds = xr.open_dataset(path, engine="netcdf4")
    lon = np.round(((ds.lon + 360) % 360).astype("float32"), decimals=4)
    lat = np.round(ds.lat.astype("float32"), decimals=4)
    ds = ds.assign_coords(lon=lon, lat=lat).sortby("lon")

    if not bool((ds.lat.diff("lat") < 0).all()):
        ds = ds.sortby("lat", ascending=False)

    return ds


def sigma_level_ids(sigma_levels: Sequence[float]) -> tuple[int, ...]:
    """Map sigma values to stable Aurora level identifiers.

    Aurora rounds level names to three decimal places when creating level-conditioned
    parameters, which collides for the highest OpenMARS sigma levels. Scaling to
    integer 1e-4 sigma values keeps the native ordering without key collisions
    while staying within Aurora's Fourier level-embedding range.
    """

    ids = tuple(int(round(float(level) * 10_000)) for level in sigma_levels)
    if len(set(ids)) != len(ids):
        raise ValueError("OpenMARS sigma levels are not unique after integer scaling.")
    return ids


def datetime_from_openmars_sol(sol: float) -> datetime:
    """Convert OpenMARS sol values to a monotonic datetime for Aurora embeddings."""

    return (OPENMARS_TIME_EPOCH + timedelta(seconds=float(sol) * MARTIAN_SOL_SECONDS)).replace(
        tzinfo=None
    )


def _as_float_tensor(array: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def _batch_from_dataset(
    ds: xr.Dataset,
    input_indices: Sequence[int],
    *,
    target_index: int | None = None,
) -> Batch:
    time_indices = list(input_indices) if target_index is None else [target_index]
    history_time = len(time_indices)
    sigma_levels = tuple(float(v) for v in ds.lev.values)
    level_ids = sigma_level_ids(sigma_levels)

    surf_vars = {
        name: _as_float_tensor(ds[name].isel(time=time_indices).values).unsqueeze(0)
        for name in SURF_VARS
    }
    atmos_vars = {
        name: _as_float_tensor(ds[name].isel(time=time_indices).values).unsqueeze(0)
        for name in ATMOS_VARS
    }

    if target_index is not None:
        # Target batches use a single history slot so their shapes match Aurora predictions.
        assert history_time == 1

    h = int(ds.sizes["lat"])
    w = int(ds.sizes["lon"])
    static_vars = {name: torch.zeros(h, w, dtype=torch.float32) for name in MARS_STATIC_VARS}
    metadata_time_index = time_indices[-1]

    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=_as_float_tensor(ds.lat.values),
            lon=_as_float_tensor(ds.lon.values),
            time=(datetime_from_openmars_sol(float(ds.time.isel(time=metadata_time_index))),),
            atmos_levels=level_ids,
        ),
    )


class OpenMARSDataset(Dataset[tuple[Batch, Batch]]):
    """PyTorch dataset yielding Aurora input/target pairs from OpenMARS files."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        history_size: int = 2,
        lead_steps: int = 1,
        target_mars_years: Sequence[int] | None = None,
        dataset_cache_size: int = 8,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1.")
        if lead_steps < 1:
            raise ValueError("lead_steps must be at least 1.")

        self.paths = tuple(Path(p) for p in paths)
        if not self.paths:
            raise ValueError("At least one OpenMARS file is required.")

        self.history_size = history_size
        self.lead_steps = lead_steps
        self.dataset_cache_size = dataset_cache_size
        self.target_mars_years = (
            frozenset(int(year) for year in target_mars_years)
            if target_mars_years is not None
            else None
        )
        self._index: list[tuple[int, int]] = []
        self._dataset_cache: OrderedDict[Path, xr.Dataset] = OrderedDict()

        for file_i, path in enumerate(self.paths):
            with open_openmars(path) as ds:
                num_times = int(ds.sizes["time"])
                mars_years = ds.MY.values.astype(int) if self.target_mars_years else None
            last_start = num_times - history_size - lead_steps
            for start in range(last_start + 1):
                target_index = start + history_size - 1 + lead_steps
                if (
                    self.target_mars_years
                    and int(mars_years[target_index]) not in self.target_mars_years
                ):
                    continue
                self._index.append((file_i, start))

    @property
    def sigma_levels(self) -> tuple[float, ...]:
        with open_openmars(self.paths[0]) as ds:
            return tuple(float(v) for v in ds.lev.values)

    @property
    def level_ids(self) -> tuple[int, ...]:
        return sigma_level_ids(self.sigma_levels)

    def __len__(self) -> int:
        return len(self._index)

    def _get_dataset(self, path: Path) -> xr.Dataset:
        if self.dataset_cache_size <= 0:
            return open_openmars(path)

        ds = self._dataset_cache.get(path)
        if ds is not None:
            self._dataset_cache.move_to_end(path)
            return ds

        ds = open_openmars(path)
        self._dataset_cache[path] = ds
        self._dataset_cache.move_to_end(path)
        while len(self._dataset_cache) > self.dataset_cache_size:
            _, old_ds = self._dataset_cache.popitem(last=False)
            old_ds.close()
        return ds

    def close(self) -> None:
        for ds in self._dataset_cache.values():
            ds.close()
        self._dataset_cache.clear()

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, index: int) -> tuple[Batch, Batch]:
        file_i, start = self._index[index]
        path = self.paths[file_i]
        input_indices = list(range(start, start + self.history_size))
        target_index = start + self.history_size - 1 + self.lead_steps

        if self.dataset_cache_size <= 0:
            with open_openmars(path) as ds:
                batch = _batch_from_dataset(ds, input_indices)
                target = _batch_from_dataset(ds, (), target_index=target_index)
        else:
            ds = self._get_dataset(path)
            batch = _batch_from_dataset(ds, input_indices)
            target = _batch_from_dataset(ds, (), target_index=target_index)

        return batch, target


class OpenMARSRolloutDataset(Dataset[tuple[Batch, list[Batch]]]):
    """OpenMARS samples for autoregressive rollout evaluation."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        history_size: int = 2,
        rollout_steps: int = 1,
        target_mars_years: Sequence[int] | None = None,
        dataset_cache_size: int = 8,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1.")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1.")

        self.paths = tuple(Path(p) for p in paths)
        if not self.paths:
            raise ValueError("At least one OpenMARS file is required.")

        self.history_size = history_size
        self.rollout_steps = rollout_steps
        self.dataset_cache_size = dataset_cache_size
        self.target_mars_years = (
            frozenset(int(year) for year in target_mars_years)
            if target_mars_years is not None
            else None
        )
        self._index: list[tuple[int, int]] = []
        self._dataset_cache: OrderedDict[Path, xr.Dataset] = OrderedDict()

        for file_i, path in enumerate(self.paths):
            with open_openmars(path) as ds:
                num_times = int(ds.sizes["time"])
                mars_years = ds.MY.values.astype(int) if self.target_mars_years else None
            last_start = num_times - history_size - rollout_steps
            for start in range(last_start + 1):
                target_indices = range(start + history_size, start + history_size + rollout_steps)
                if self.target_mars_years and any(
                    int(mars_years[i]) not in self.target_mars_years for i in target_indices
                ):
                    continue
                self._index.append((file_i, start))

    @property
    def sigma_levels(self) -> tuple[float, ...]:
        with open_openmars(self.paths[0]) as ds:
            return tuple(float(v) for v in ds.lev.values)

    @property
    def level_ids(self) -> tuple[int, ...]:
        return sigma_level_ids(self.sigma_levels)

    def __len__(self) -> int:
        return len(self._index)

    def _get_dataset(self, path: Path) -> xr.Dataset:
        if self.dataset_cache_size <= 0:
            return open_openmars(path)

        ds = self._dataset_cache.get(path)
        if ds is not None:
            self._dataset_cache.move_to_end(path)
            return ds

        ds = open_openmars(path)
        self._dataset_cache[path] = ds
        self._dataset_cache.move_to_end(path)
        while len(self._dataset_cache) > self.dataset_cache_size:
            _, old_ds = self._dataset_cache.popitem(last=False)
            old_ds.close()
        return ds

    def close(self) -> None:
        for ds in self._dataset_cache.values():
            ds.close()
        self._dataset_cache.clear()

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, index: int) -> tuple[Batch, list[Batch]]:
        file_i, start = self._index[index]
        path = self.paths[file_i]
        input_indices = list(range(start, start + self.history_size))
        target_indices = list(
            range(start + self.history_size, start + self.history_size + self.rollout_steps)
        )

        if self.dataset_cache_size <= 0:
            with open_openmars(path) as ds:
                batch = _batch_from_dataset(ds, input_indices)
                targets = [
                    _batch_from_dataset(ds, (), target_index=target_index)
                    for target_index in target_indices
                ]
        else:
            ds = self._get_dataset(path)
            batch = _batch_from_dataset(ds, input_indices)
            targets = [
                _batch_from_dataset(ds, (), target_index=target_index)
                for target_index in target_indices
            ]

        return batch, targets


def collate_batches(batches: Sequence[Batch]) -> Batch:
    """Collate Aurora batches along the batch dimension."""

    if not batches:
        raise ValueError("Cannot collate an empty batch list.")

    first = batches[0]
    return Batch(
        surf_vars={
            name: torch.cat([batch.surf_vars[name] for batch in batches], dim=0)
            for name in first.surf_vars
        },
        static_vars=first.static_vars,
        atmos_vars={
            name: torch.cat([batch.atmos_vars[name] for batch in batches], dim=0)
            for name in first.atmos_vars
        },
        metadata=Metadata(
            lat=first.metadata.lat,
            lon=first.metadata.lon,
            time=tuple(time for batch in batches for time in batch.metadata.time),
            atmos_levels=first.metadata.atmos_levels,
            rollout_step=first.metadata.rollout_step,
        ),
    )


def collate_batch_pairs(pairs: Sequence[tuple[Batch, Batch]]) -> tuple[Batch, Batch]:
    inputs, targets = zip(*pairs)
    return collate_batches(inputs), collate_batches(targets)


def collate_rollout_samples(
    samples: Sequence[tuple[Batch, list[Batch]]],
) -> tuple[Batch, list[Batch]]:
    inputs, target_lists = zip(*samples)
    rollout_steps = len(target_lists[0])
    return (
        collate_batches(inputs),
        [
            collate_batches([target_list[step] for target_list in target_lists])
            for step in range(rollout_steps)
        ],
    )


def _accumulate_stats(values: np.ndarray, sum_: np.ndarray, sumsq: np.ndarray, count: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    return (
        sum_ + np.where(mask, values, 0.0).sum(axis=0),
        sumsq + np.where(mask, values * values, 0.0).sum(axis=0),
        count + mask.sum(axis=0),
    )


def compute_openmars_stats(
    paths: Iterable[str | Path],
    *,
    max_time_steps: int | None = None,
    mars_years: Sequence[int] | None = None,
    min_scale: float = 1e-6,
) -> OpenMARSStats:
    """Compute per-variable OpenMARS mean/std statistics."""

    paths = tuple(Path(p) for p in paths)
    if not paths:
        raise ValueError("At least one OpenMARS file is required.")
    mars_years_set = frozenset(int(year) for year in mars_years) if mars_years else None

    locations_out: dict[str, float] = {"mars_static": 0.0}
    scales_out: dict[str, float] = {"mars_static": 1.0}
    sigma_levels: tuple[float, ...] | None = None
    level_ids: tuple[int, ...] | None = None

    surf_sum = {name: np.array(0.0) for name in SURF_VARS}
    surf_sumsq = {name: np.array(0.0) for name in SURF_VARS}
    surf_count = {name: np.array(0, dtype=np.int64) for name in SURF_VARS}
    atmos_sum: dict[str, np.ndarray] = {}
    atmos_sumsq: dict[str, np.ndarray] = {}
    atmos_count: dict[str, np.ndarray] = {}

    for path in paths:
        with open_openmars(path) as ds:
            if sigma_levels is None:
                sigma_levels = tuple(float(v) for v in ds.lev.values)
                level_ids = sigma_level_ids(sigma_levels)
                for name in ATMOS_VARS:
                    atmos_sum[name] = np.zeros(len(level_ids), dtype=np.float64)
                    atmos_sumsq[name] = np.zeros(len(level_ids), dtype=np.float64)
                    atmos_count[name] = np.zeros(len(level_ids), dtype=np.int64)
            elif tuple(float(v) for v in ds.lev.values) != sigma_levels:
                raise ValueError(f"Sigma levels in {path} do not match previous files.")

            time_indices = np.arange(int(ds.sizes["time"]))
            if max_time_steps is not None:
                time_indices = time_indices[:max_time_steps]
            if mars_years_set:
                my_values = ds.MY.isel(time=time_indices).values.astype(int)
                time_indices = time_indices[np.isin(my_values, list(mars_years_set))]
            if len(time_indices) == 0:
                continue

            for name in SURF_VARS:
                values = ds[name].isel(time=time_indices).values.reshape(-1)
                surf_sum[name], surf_sumsq[name], surf_count[name] = _accumulate_stats(
                    values, surf_sum[name], surf_sumsq[name], surf_count[name]
                )

            for name in ATMOS_VARS:
                values = ds[name].isel(time=time_indices).values
                # Reduce over time, latitude, and longitude, preserving level.
                values = np.moveaxis(values, 1, -1).reshape(-1, len(level_ids))
                atmos_sum[name], atmos_sumsq[name], atmos_count[name] = _accumulate_stats(
                    values, atmos_sum[name], atmos_sumsq[name], atmos_count[name]
                )

    assert sigma_levels is not None
    assert level_ids is not None

    for name in SURF_VARS:
        if int(surf_count[name]) == 0:
            raise ValueError(f"No finite values found for {name}.")
        mean = float(surf_sum[name] / surf_count[name])
        var = max(float(surf_sumsq[name] / surf_count[name] - mean * mean), 0.0)
        locations_out[name] = mean
        scales_out[name] = max(var**0.5, min_scale)

    for name in ATMOS_VARS:
        for i, level_id in enumerate(level_ids):
            if int(atmos_count[name][i]) == 0:
                raise ValueError(f"No finite values found for {name} level {level_id}.")
            mean = float(atmos_sum[name][i] / atmos_count[name][i])
            var = max(float(atmos_sumsq[name][i] / atmos_count[name][i] - mean * mean), 0.0)
            key = f"{name}_{level_to_str(level_id)}"
            locations_out[key] = mean
            scales_out[key] = max(var**0.5, min_scale)

    return OpenMARSStats(
        locations=locations_out,
        scales=scales_out,
        sigma_levels=sigma_levels,
        level_ids=level_ids,
    )


def register_openmars_stats(stats: OpenMARSStats) -> None:
    """Register OpenMARS normalisation stats in Aurora's global dictionaries."""

    locations.update(stats.locations)
    scales.update(stats.scales)


def save_openmars_stats(stats: OpenMARSStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_json_dict(), indent=2, sort_keys=True) + "\n")


def load_openmars_stats(path: str | Path, *, register: bool = True) -> OpenMARSStats:
    stats = OpenMARSStats.from_json_dict(json.loads(Path(path).read_text()))
    if register:
        register_openmars_stats(stats)
    return stats
