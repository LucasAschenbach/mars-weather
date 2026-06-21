#!/usr/bin/env python3
"""Evaluate recursive Mars Aurora rollouts on an OpenMARS split."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mars_weather import (
    ATMOS_VARS,
    SURF_VARS,
    collate_rollout_samples,
    load_openmars_stats,
    load_split_manifest,
    make_mars_aurora,
    normalized_mse_losses,
    register_openmars_stats,
    rollout_dataset_from_manifest,
)
from mars_weather.openmars import OpenMARSStats
from aurora.normalisation import level_to_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        choices=("none", "persistence", "climatology"),
        default="none",
        help="Evaluate a no-learned-model baseline instead of a checkpoint.",
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--model-size", choices=("base", "small"), default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--dataset-cache-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def load_run_config(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def resolve_paths(args: argparse.Namespace, config: dict) -> tuple[Path, Path, Path]:
    if args.baseline != "none":
        checkpoint = Path(f"<{args.baseline}>")
    elif args.checkpoint is not None:
        checkpoint = args.checkpoint
    elif args.run_dir is not None:
        checkpoint = args.run_dir / "checkpoint_latest.pt"
    else:
        raise SystemExit("Provide --run-dir or --checkpoint.")

    split_manifest = args.split_manifest
    if split_manifest is None and config.get("split_manifest"):
        split_manifest = Path(config["split_manifest"])
    if split_manifest is None:
        raise SystemExit("Provide --split-manifest or evaluate from a run with config.json.")

    if args.output is not None:
        output = args.output
    elif args.run_dir is not None:
        prefix = f"{args.baseline}_" if args.baseline != "none" else ""
        output = args.run_dir / f"{prefix}rollout_eval_{args.split}_{args.rollout_steps}step.json"
    else:
        output = Path("artifacts") / f"{args.baseline}_rollout_eval_{args.split}_{args.rollout_steps}step.json"

    return checkpoint, split_manifest, output


def load_stats(args: argparse.Namespace, checkpoint_data: dict, config: dict) -> OpenMARSStats:
    if "stats" in checkpoint_data:
        stats = OpenMARSStats.from_json_dict(checkpoint_data["stats"])
        register_openmars_stats(stats)
        return stats

    stats_path = args.stats
    if stats_path is None and config.get("stats"):
        stats_path = Path(config["stats"])
    if stats_path is None:
        raise SystemExit("Checkpoint has no stats. Provide --stats.")
    return load_openmars_stats(stats_path, register=True)


def add_tensor_metric(
    accum: dict[str, dict[str, float]],
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    diff = (pred - target).detach().float()
    item = accum.setdefault(
        name,
        {
            "sse": 0.0,
            "sae": 0.0,
            "target_sumsq": 0.0,
            "target_abs_sum": 0.0,
            "count": 0.0,
            "window_rmse": [],
            "window_relative_rmse": [],
        },
    )
    item["sse"] += float(torch.sum(diff * diff).cpu())
    item["sae"] += float(torch.sum(torch.abs(diff)).cpu())
    item["target_sumsq"] += float(torch.sum(target.detach().float() * target.detach().float()).cpu())
    item["target_abs_sum"] += float(torch.sum(torch.abs(target.detach().float())).cpu())
    item["count"] += float(diff.numel())

    per_window_diff = diff.reshape(diff.shape[0], -1)
    per_window_target = target.detach().float().reshape(target.shape[0], -1)
    window_rmse = torch.sqrt(torch.mean(per_window_diff * per_window_diff, dim=1))
    target_rms = torch.sqrt(torch.mean(per_window_target * per_window_target, dim=1))
    window_relative_rmse = window_rmse / torch.clamp(target_rms, min=1e-8)
    item["window_rmse"].extend(float(v) for v in window_rmse.cpu())
    item["window_relative_rmse"].extend(float(v) for v in window_relative_rmse.cpu())


def summarize_window_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(torch.mean(tensor)),
        "std": float(torch.std(tensor, unbiased=False)) if len(values) > 1 else 0.0,
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
    }


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


def persistence_prediction(batch, target):
    return dataclasses.replace(
        target,
        surf_vars={name: batch.surf_vars[name][:, -1:].clone() for name in SURF_VARS},
        atmos_vars={name: batch.atmos_vars[name][:, -1:].clone() for name in ATMOS_VARS},
    )


def climatology_prediction(target, stats: OpenMARSStats):
    surf_vars = {
        name: torch.full_like(target.surf_vars[name], float(stats.locations[name]))
        for name in SURF_VARS
    }
    atmos_vars = {}
    for name in ATMOS_VARS:
        values = torch.empty_like(target.atmos_vars[name])
        for level_i, level_id in enumerate(stats.level_ids):
            key = f"{name}_{level_to_str(level_id)}"
            values[:, :, level_i] = float(stats.locations[key])
        atmos_vars[name] = values
    return dataclasses.replace(target, surf_vars=surf_vars, atmos_vars=atmos_vars)


def main() -> None:
    args = parse_args()
    config = load_run_config(args.run_dir)
    checkpoint_path, split_manifest, output_path = resolve_paths(args, config)

    checkpoint = {}
    if args.baseline == "none":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = load_stats(args, checkpoint, config)

    model_size = args.model_size or config.get("model_size") or "base"
    model = None
    if args.baseline == "none":
        model = make_mars_aurora(
            level_ids=stats.level_ids,
            size=model_size,
            load_checkpoint=False,
            autocast=not bool(config.get("no_aurora_autocast", False)),
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to(args.device)

    manifest = load_split_manifest(split_manifest)
    dataset = rollout_dataset_from_manifest(
        manifest,
        args.split,
        rollout_steps=args.rollout_steps,
        dataset_cache_size=args.dataset_cache_size,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_rollout_samples,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    norm_sums = {
        step: {name: 0.0 for name in SURF_VARS + ATMOS_VARS}
        for step in range(1, args.rollout_steps + 1)
    }
    norm_count = {step: 0.0 for step in range(1, args.rollout_steps + 1)}
    physical: dict[int, dict[str, dict[str, float]]] = {
        step: {} for step in range(1, args.rollout_steps + 1)
    }
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_i, (batch, targets) in enumerate(tqdm(loader, desc=f"rollout {args.split}")):
            if args.max_batches is not None and batch_i >= args.max_batches:
                break
            batch = batch.to(args.device)
            for step, target in enumerate(targets, start=1):
                target = target.to(args.device)
                if args.baseline == "persistence":
                    pred = persistence_prediction(batch, target)
                elif args.baseline == "climatology":
                    pred = climatology_prediction(target, stats)
                else:
                    assert model is not None
                    pred = model(batch)
                losses = normalized_mse_losses(pred, target)
                batch_size = next(iter(target.surf_vars.values())).shape[0]
                for name, value in losses.items():
                    norm_sums[step][name] += float(value.detach().cpu()) * batch_size
                norm_count[step] += batch_size

                for name in SURF_VARS:
                    add_tensor_metric(
                        physical[step],
                        name,
                        pred.surf_vars[name],
                        target.surf_vars[name],
                    )
                for name in ATMOS_VARS:
                    add_tensor_metric(
                        physical[step],
                        name,
                        pred.atmos_vars[name],
                        target.atmos_vars[name],
                    )
                batch = advance_batch_history(batch, pred)

    lead_metrics = {}
    for step in range(1, args.rollout_steps + 1):
        norm_mse = {
            name: value / max(norm_count[step], 1.0)
            for name, value in norm_sums[step].items()
        }
        physical_metrics = {
            name: {
                "rmse": (values["sse"] / max(values["count"], 1.0)) ** 0.5,
                "mae": values["sae"] / max(values["count"], 1.0),
                "relative_rmse": (
                    (values["sse"] / max(values["count"], 1.0)) ** 0.5
                    / max((values["target_sumsq"] / max(values["count"], 1.0)) ** 0.5, 1e-8)
                ),
                "relative_mae": (
                    values["sae"]
                    / max(values["count"], 1.0)
                    / max(values["target_abs_sum"] / max(values["count"], 1.0), 1e-8)
                ),
                "window_rmse": summarize_window_values(values["window_rmse"]),
                "window_relative_rmse": summarize_window_values(values["window_relative_rmse"]),
                "count": values["count"],
            }
            for name, values in physical[step].items()
        }
        lead_metrics[str(step)] = {
            "normalized_mse": norm_mse,
            "normalized_mse_mean": sum(norm_mse.values()) / max(len(norm_mse), 1),
            "physical": physical_metrics,
            "num_windows": int(norm_count[step]),
        }

    metrics = {
        "checkpoint": str(checkpoint_path),
        "baseline": args.baseline,
        "split_manifest": str(split_manifest),
        "split": args.split,
        "model_size": model_size,
        "rollout_steps": args.rollout_steps,
        "num_windows": int(norm_count[1]) if norm_count else 0,
        "max_batches": args.max_batches,
        "elapsed_sec": time.perf_counter() - started,
        "lead_steps": lead_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    csv_path = None
    if args.write_csv:
        csv_path = output_path.with_suffix(".csv")
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lead_step", "variable", "normalized_mse", "rmse", "mae", "count"])
            for step, step_metrics in lead_metrics.items():
                for name in SURF_VARS + ATMOS_VARS:
                    writer.writerow(
                        [
                            step,
                            name,
                            step_metrics["normalized_mse"][name],
                            step_metrics["physical"][name]["rmse"],
                            step_metrics["physical"][name]["mae"],
                            int(step_metrics["physical"][name]["count"]),
                        ]
                    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "csv": str(csv_path) if csv_path else None,
                "rollout_steps": args.rollout_steps,
                "final_step_normalized_mse_mean": lead_metrics[str(args.rollout_steps)][
                    "normalized_mse_mean"
                ],
                "num_windows": metrics["num_windows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
