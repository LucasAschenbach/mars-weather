#!/usr/bin/env python3
"""Export one recursive OpenMARS rollout and matching truth as NetCDF files."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import xarray as xr

from mars_weather import (
    ATMOS_VARS,
    SURF_VARS,
    load_openmars_stats,
    load_split_manifest,
    make_mars_aurora,
    register_openmars_stats,
    rollout_dataset_from_manifest,
)
from mars_weather.openmars import open_openmars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-run-dir", type=Path, required=True)
    parser.add_argument("--random-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=Path("artifacts/openmars_stats.json"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=84)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/app_rollouts"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def resolve_checkpoint(run_dir: Path) -> Path:
    latest = run_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest
    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return checkpoints[-1]


def load_model(run_dir: Path, *, level_ids: tuple[int, ...], device: torch.device):
    config = load_run_config(run_dir)
    checkpoint_path = resolve_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_mars_aurora(
        level_ids=level_ids,
        size=config.get("model_size", "base"),
        load_checkpoint=False,
        autocast=not bool(config.get("no_aurora_autocast", False)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return model, checkpoint_path


def advance_batch_history(batch, pred):
    return dataclasses.replace(
        pred,
        surf_vars={
            name: torch.cat([batch.surf_vars[name][:, 1:], value], dim=1)
            for name, value in pred.surf_vars.items()
        },
        atmos_vars={
            name: torch.cat([batch.atmos_vars[name][:, 1:], value], dim=1)
            for name, value in pred.atmos_vars.items()
        },
    )


def batch_to_frames(batch) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in SURF_VARS:
        arrays[name] = batch.surf_vars[name][0, -1].detach().cpu().numpy()
    for name in ATMOS_VARS:
        arrays[name] = batch.atmos_vars[name][0, -1].detach().cpu().numpy()
    return arrays


def append_batch_frame(frames: dict[str, list[np.ndarray]], batch) -> None:
    arrays = batch_to_frames(batch)
    for name, array in arrays.items():
        frames[name].append(np.asarray(array, dtype=np.float32))


def rollout_model(model, initial_batch, targets, *, device: torch.device) -> dict[str, list[np.ndarray]]:
    frames: dict[str, list[np.ndarray]] = {name: [] for name in SURF_VARS + ATMOS_VARS}
    batch = initial_batch.to(device)
    append_batch_frame(frames, batch)
    with torch.inference_mode():
        for _target in targets:
            pred = model(batch)
            append_batch_frame(frames, pred)
            batch = advance_batch_history(batch, pred)
    return frames


def truth_frames(initial_batch, targets) -> dict[str, list[np.ndarray]]:
    frames: dict[str, list[np.ndarray]] = {name: [] for name in SURF_VARS + ATMOS_VARS}
    append_batch_frame(frames, initial_batch)
    for target in targets:
        append_batch_frame(frames, target)
    return frames


def sample_time_metadata(dataset, sample_index: int, rollout_steps: int) -> dict[str, list[float] | list[int]]:
    file_i, start = dataset._index[sample_index]
    path = dataset.paths[file_i]
    history_size = dataset.history_size
    time_indices = [start + history_size - 1 + step for step in range(rollout_steps + 1)]
    with open_openmars(path) as ds:
        data: dict[str, list[float] | list[int]] = {
            "openmars_time": [float(v) for v in ds.time.isel(time=time_indices).values],
        }
        if "Ls" in ds:
            data["Ls"] = [float(v) for v in ds.Ls.isel(time=time_indices).values]
        if "MY" in ds:
            data["MY"] = [int(v) for v in ds.MY.isel(time=time_indices).values]
    return data


def frames_to_dataset(
    frames: dict[str, list[np.ndarray]],
    *,
    initial_batch,
    sigma_levels: tuple[float, ...],
    time_metadata: dict[str, list[float] | list[int]],
    source: str,
    checkpoint: Path | None,
) -> xr.Dataset:
    lead_hours = np.arange(len(frames[SURF_VARS[0]]), dtype=np.int32) * 2
    coords = {
        "time": lead_hours,
        "lat": initial_batch.metadata.lat.detach().cpu().numpy().astype(np.float32),
        "lon": initial_batch.metadata.lon.detach().cpu().numpy().astype(np.float32),
        "lev": np.asarray(sigma_levels, dtype=np.float32),
    }
    data_vars = {}
    for name in SURF_VARS:
        data_vars[name] = (("time", "lat", "lon"), np.stack(frames[name], axis=0))
    for name in ATMOS_VARS:
        data_vars[name] = (("time", "lev", "lat", "lon"), np.stack(frames[name], axis=0))
    if "Ls" in time_metadata:
        data_vars["Ls"] = ("time", np.asarray(time_metadata["Ls"], dtype=np.float32))
    if "MY" in time_metadata:
        data_vars["MY"] = ("time", np.asarray(time_metadata["MY"], dtype=np.int32))
    if "openmars_time" in time_metadata:
        data_vars["openmars_time"] = (
            "time",
            np.asarray(time_metadata["openmars_time"], dtype=np.float64),
        )

    base_time = initial_batch.metadata.time[-1]
    if base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=timezone.utc)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs.update(
        {
            "source": source,
            "base_time": base_time.isoformat(),
            "step_hours": 2,
            "checkpoint": str(checkpoint) if checkpoint else "",
        }
    )
    return ds


def write_dataset(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 3} for name in ds.data_vars}
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    pretrained_config = load_run_config(args.pretrained_run_dir)
    split_manifest = args.split_manifest
    if split_manifest is None:
        split_manifest = Path(pretrained_config.get("split_manifest", ""))
    if not split_manifest:
        raise SystemExit("Provide --split-manifest or use a run dir with config.json.")

    stats_path = args.stats
    if not stats_path.exists() and pretrained_config.get("stats"):
        stats_path = Path(pretrained_config["stats"])
    stats = load_openmars_stats(stats_path, register=False)
    register_openmars_stats(stats)

    manifest = load_split_manifest(split_manifest)
    dataset = rollout_dataset_from_manifest(
        manifest,
        args.split,
        rollout_steps=args.rollout_steps,
        dataset_cache_size=4,
    )
    initial_batch, targets = dataset[args.sample_index]
    time_metadata = sample_time_metadata(dataset, args.sample_index, args.rollout_steps)

    pretrained_model, pretrained_checkpoint = load_model(
        args.pretrained_run_dir,
        level_ids=stats.level_ids,
        device=device,
    )
    random_model, random_checkpoint = load_model(
        args.random_run_dir,
        level_ids=stats.level_ids,
        device=device,
    )

    outputs = {
        "pretrained": frames_to_dataset(
            rollout_model(pretrained_model, initial_batch, targets, device=device),
            initial_batch=initial_batch,
            sigma_levels=stats.sigma_levels,
            time_metadata=time_metadata,
            source="pretrained",
            checkpoint=pretrained_checkpoint,
        ),
        "random-init": frames_to_dataset(
            rollout_model(random_model, initial_batch, targets, device=device),
            initial_batch=initial_batch,
            sigma_levels=stats.sigma_levels,
            time_metadata=time_metadata,
            source="random-init",
            checkpoint=random_checkpoint,
        ),
        "ground-truth": frames_to_dataset(
            truth_frames(initial_batch, targets),
            initial_batch=initial_batch,
            sigma_levels=stats.sigma_levels,
            time_metadata=time_metadata,
            source="ground-truth",
            checkpoint=None,
        ),
    }

    written = {}
    for name, ds in outputs.items():
        path = args.output_dir / f"{name}_7day_rollout.nc"
        write_dataset(ds, path)
        written[name] = str(path)
    print(json.dumps({"outputs": written, "frames": args.rollout_steps + 1}, indent=2))


if __name__ == "__main__":
    main()
