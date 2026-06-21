#!/usr/bin/env python3
"""Plot recursive rollout evaluation metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

_matplotlib_config = Path("/tmp/mars_weather_matplotlib")
_matplotlib_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_config))

import matplotlib.pyplot as plt


SURFACE_VARS = ("ps", "tsurf", "co2ice", "dustcol")
ATMOS_VARS = ("u", "v", "temp")
DISPLAY_NAMES = {
    "ps": "Surface pressure",
    "tsurf": "Surface temperature",
    "co2ice": "CO2 ice",
    "dustcol": "Dust column",
    "u": "Zonal wind",
    "v": "Meridional wind",
    "temp": "Atmospheric temperature",
}
UNITS = {
    "ps": "Pa",
    "tsurf": "K",
    "co2ice": "kg m^-2",
    "dustcol": "opacity",
    "u": "m s^-1",
    "v": "m s^-1",
    "temp": "K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_json", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels, one per eval JSON.",
    )
    parser.add_argument(
        "--summary-metric",
        choices=("relative_rmse", "normalized_rmse", "normalized_mse"),
        default="relative_rmse",
        help="Metric for the two-panel summary plot.",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def lead_steps(metrics: dict) -> list[int]:
    return sorted(int(step) for step in metrics["lead_steps"])


def values_for(metrics: dict, variable: str, metric: str) -> list[float]:
    values = []
    for step in lead_steps(metrics):
        step_metrics = metrics["lead_steps"][str(step)]
        if metric == "rmse":
            values.append(float(step_metrics["physical"][variable]["rmse"]))
        elif metric == "relative_rmse":
            physical = step_metrics["physical"][variable]
            if "relative_rmse" in physical:
                values.append(float(physical["relative_rmse"]))
            else:
                values.append(float(step_metrics["normalized_mse"][variable]) ** 0.5)
        elif metric == "mae":
            values.append(float(step_metrics["physical"][variable]["mae"]))
        elif metric == "normalized_rmse":
            values.append(float(step_metrics["normalized_mse"][variable]) ** 0.5)
        elif metric == "normalized_mse":
            values.append(float(step_metrics["normalized_mse"][variable]))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return values


def window_band_for(metrics: dict, variable: str, metric: str) -> tuple[list[float], list[float]] | None:
    means = []
    stds = []
    for step in lead_steps(metrics):
        physical = metrics["lead_steps"][str(step)]["physical"][variable]
        key = "window_rmse" if metric == "rmse" else "window_relative_rmse"
        if key not in physical:
            return None
        means.append(float(physical[key]["mean"]))
        stds.append(float(physical[key]["std"]))
    return means, stds


def eval_label(path: Path, metrics: dict) -> str:
    baseline = metrics.get("baseline")
    if baseline and baseline != "none":
        return str(baseline)
    run_name = path.parent.name
    if "random_init" in run_name:
        return "random init"
    return run_name


def common_subtitle(metrics_list: list[dict]) -> str:
    splits = sorted({str(metrics.get("split", "?")) for metrics in metrics_list})
    steps = sorted({str(metrics.get("rollout_steps", "?")) for metrics in metrics_list})
    windows = sorted({str(metrics.get("num_windows", "?")) for metrics in metrics_list})
    return f"split={','.join(splits)}, windows={','.join(windows)}, steps={','.join(steps)}"


def plot_rmse_grid(
    metrics_list: list[dict],
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    variables = SURFACE_VARS + ATMOS_VARS
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    axes_flat = axes.ravel()
    legend_handles = []

    for ax, variable in zip(axes_flat, variables):
        for line_i, (metrics, label) in enumerate(zip(metrics_list, labels)):
            steps = lead_steps(metrics)
            values = values_for(metrics, variable, "rmse")
            (line,) = ax.plot(
                steps,
                values,
                marker="o",
                linewidth=2,
                label=label,
            )
            band = window_band_for(metrics, variable, "rmse")
            if band is not None:
                means, stds = band
                lower = np.maximum(np.asarray(means) - np.asarray(stds), 0.0)
                upper = np.asarray(means) + np.asarray(stds)
                ax.fill_between(steps, lower, upper, alpha=0.14)
            if len(legend_handles) < len(labels) and line_i == len(legend_handles):
                legend_handles.append(line)
        ax.set_title(DISPLAY_NAMES.get(variable, variable))
        ax.set_xlabel("Rollout lead step")
        ax.set_ylabel(f"RMSE ({UNITS.get(variable, '')})")
        ax.grid(True, alpha=0.3)

    axes_flat[-1].axis("off")
    axes_flat[-1].legend(legend_handles, labels, loc="center")
    fig.suptitle(f"{title}\n{common_subtitle(metrics_list)}", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary(
    metrics_list: list[dict],
    labels: list[str],
    output_path: Path,
    title: str,
    metric: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    metric_label = {
        "relative_rmse": "Mean relative RMSE",
        "normalized_rmse": "Mean normalized RMSE",
        "normalized_mse": "Mean normalized MSE",
    }[metric]

    for ax, variables, panel_title in (
        (axes[0], SURFACE_VARS, "Surface Variables"),
        (axes[1], ATMOS_VARS, "Atmospheric Variables"),
    ):
        for metrics, label in zip(metrics_list, labels):
            means = []
            for step in lead_steps(metrics):
                step_values = []
                for name in variables:
                    step_metrics = metrics["lead_steps"][str(step)]
                    if metric == "relative_rmse":
                        physical = step_metrics["physical"][name]
                        if "relative_rmse" in physical:
                            step_values.append(float(physical["relative_rmse"]))
                        else:
                            step_values.append(float(step_metrics["normalized_mse"][name]) ** 0.5)
                    elif metric == "normalized_rmse":
                        step_values.append(float(step_metrics["normalized_mse"][name]) ** 0.5)
                    else:
                        step_values.append(float(step_metrics["normalized_mse"][name]))
                means.append(sum(step_values) / len(step_values))
            ax.plot(lead_steps(metrics), means, marker="o", linewidth=2, label=label)
        ax.set_title(panel_title)
        ax.set_xlabel("Rollout lead step")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"{title}\n{common_subtitle(metrics_list)}", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.eval_json):
        raise SystemExit("--labels must have the same number of values as eval JSON paths.")

    metrics_list = [load_metrics(path) for path in args.eval_json]
    labels = args.labels or [
        eval_label(path, metrics) for path, metrics in zip(args.eval_json, metrics_list)
    ]
    first_json = args.eval_json[0]
    output_dir = args.output_dir or first_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    title = args.title
    if title is None:
        title = "OpenMARS Recursive Rollout Eval"

    stem = first_json.stem if len(args.eval_json) == 1 else "rollout_eval_comparison"
    rmse_path = output_dir / f"{stem}_rmse_grid.png"
    summary_path = output_dir / f"{stem}_{args.summary_metric}_summary.png"
    plot_rmse_grid(metrics_list, labels, rmse_path, title)
    plot_summary(metrics_list, labels, summary_path, title, args.summary_metric)

    print(json.dumps({"rmse_grid": str(rmse_path), "normalized_mse_summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
