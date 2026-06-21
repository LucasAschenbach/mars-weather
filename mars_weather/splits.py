"""Reproducible OpenMARS split manifests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from mars_weather.openmars import OpenMARSDataset, OpenMARSRolloutDataset, open_openmars


@dataclasses.dataclass(frozen=True)
class OpenMARSFileSummary:
    path: str
    mars_years: tuple[int, ...]
    start_sol: float
    end_sol: float
    start_ls: float
    end_ls: float
    num_times: int

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def summarize_openmars_file(path: str | Path, *, root: str | Path | None = None) -> OpenMARSFileSummary:
    path = Path(path)
    display_path = path if root is None else path.relative_to(root)
    with open_openmars(path) as ds:
        return OpenMARSFileSummary(
            path=display_path.as_posix(),
            mars_years=tuple(sorted(set(int(v) for v in ds.MY.values.tolist()))),
            start_sol=float(ds.time.values[0]),
            end_sol=float(ds.time.values[-1]),
            start_ls=float(ds.Ls.values[0]),
            end_ls=float(ds.Ls.values[-1]),
            num_times=int(ds.sizes["time"]),
        )


def create_openmars_split_manifest(
    files: list[str | Path],
    *,
    train_years: tuple[int, ...] = (28, 29, 30, 31, 32, 33, 34),
    val_years: tuple[int, ...] = (35,),
    root: str | Path = ".",
    history_size: int = 2,
    lead_steps: int = 1,
) -> dict[str, Any]:
    """Create a deterministic target-year split manifest."""

    root_path = Path(root)
    paths = tuple(sorted(Path(path) for path in files))
    summaries = [summarize_openmars_file(path, root=root_path) for path in paths]
    train_dataset = OpenMARSDataset(
        paths,
        history_size=history_size,
        lead_steps=lead_steps,
        target_mars_years=train_years,
    )
    val_dataset = OpenMARSDataset(
        paths,
        history_size=history_size,
        lead_steps=lead_steps,
        target_mars_years=val_years,
    )

    return {
        "format": "openmars-target-year-split-v1",
        "history_size": history_size,
        "lead_steps": lead_steps,
        "assignment": "Samples are assigned by the Mars Year of the target frame.",
        "splits": {
            "train": {
                "target_mars_years": list(train_years),
                "num_samples": len(train_dataset),
            },
            "val": {
                "target_mars_years": list(val_years),
                "num_samples": len(val_dataset),
            },
        },
        "files": [summary.to_json_dict() for summary in summaries],
    }


def save_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def dataset_from_manifest(
    manifest: dict[str, Any],
    split: str,
    *,
    root: str | Path = ".",
    dataset_cache_size: int = 8,
) -> OpenMARSDataset:
    split_data = manifest["splits"][split]
    paths = [Path(root) / file_info["path"] for file_info in manifest["files"]]
    return OpenMARSDataset(
        paths,
        history_size=int(manifest["history_size"]),
        lead_steps=int(manifest["lead_steps"]),
        target_mars_years=tuple(int(year) for year in split_data["target_mars_years"]),
        dataset_cache_size=dataset_cache_size,
    )


def rollout_dataset_from_manifest(
    manifest: dict[str, Any],
    split: str,
    *,
    rollout_steps: int,
    root: str | Path = ".",
    dataset_cache_size: int = 8,
) -> OpenMARSRolloutDataset:
    split_data = manifest["splits"][split]
    paths = [Path(root) / file_info["path"] for file_info in manifest["files"]]
    return OpenMARSRolloutDataset(
        paths,
        history_size=int(manifest["history_size"]),
        rollout_steps=rollout_steps,
        target_mars_years=tuple(int(year) for year in split_data["target_mars_years"]),
        dataset_cache_size=dataset_cache_size,
    )
