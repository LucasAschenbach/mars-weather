#!/usr/bin/env python3
"""Export model-only OpenMARS Aurora weights for Hugging Face upload."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from safetensors.torch import save_file as save_safetensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("safetensors", "torch"), default="safetensors")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--description", default="")
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_model_card(
    path: Path,
    *,
    model_name: str,
    checkpoint: Path,
    run_config: dict | None,
    description: str,
) -> None:
    training_summary = ""
    if run_config:
        training_summary = "\n".join(
            [
                f"- Model size: `{run_config.get('model_size', 'unknown')}`",
                f"- Split manifest: `{run_config.get('split_manifest', 'unknown')}`",
                f"- Epochs: `{run_config.get('epochs', 'unknown')}`",
                f"- Batch size per GPU: `{run_config.get('batch_size', 'unknown')}`",
                f"- Base LR: `{run_config.get('base_lr', 'unknown')}`",
                f"- New/Mars LR: `{run_config.get('new_lr', 'unknown')}`",
                f"- Resume checkpoint: `{run_config.get('resume', 'none')}`",
            ]
        )

    text = f"""---
library_name: pytorch
tags:
- mars
- weather
- aurora
- openmars
---

# {model_name}

{description}

This repository contains model-only weights exported from an OpenMARS fine-tuned
Aurora checkpoint. Optimizer state and RNG state were intentionally stripped.

## Source checkpoint

`{checkpoint}`

## Training configuration

{training_summary or "See `training_config.json` if present."}

## Loading

Use this repository with the `mars_weather.make_mars_aurora` helper from the
project code. Load `openmars_stats.json`, construct the configured Mars Aurora
model, and then load `model.safetensors` or `pytorch_model.bin` as a state dict.
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    stats = checkpoint.get("stats")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "safetensors":
        weights_path = args.output_dir / "model.safetensors"
        save_safetensors(state_dict, str(weights_path))
    else:
        weights_path = args.output_dir / "pytorch_model.bin"
        torch.save(state_dict, weights_path)

    if stats is not None:
        (args.output_dir / "openmars_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n"
        )

    export_config = {
        "format": "mars-weather-aurora-weights-v1",
        "model_name": args.model_name,
        "weights_file": weights_path.name,
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step_in_epoch": checkpoint.get("step_in_epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_distributed": checkpoint.get("distributed"),
        "has_openmars_stats": stats is not None,
    }
    run_config = load_json(args.run_dir / "config.json") if args.run_dir else None
    if run_config is not None:
        export_config["training_config_file"] = "training_config.json"
        (args.output_dir / "training_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n"
        )

    metadata_path = args.run_dir / "metadata.json" if args.run_dir else None
    if metadata_path and metadata_path.exists():
        shutil.copy2(metadata_path, args.output_dir / "training_metadata.json")

    eval_path = args.run_dir / "rollout_eval_val_20step.json" if args.run_dir else None
    if eval_path and eval_path.exists():
        shutil.copy2(eval_path, args.output_dir / "rollout_eval_val_20step.json")

    (args.output_dir / "config.json").write_text(
        json.dumps(export_config, indent=2, sort_keys=True) + "\n"
    )
    write_model_card(
        args.output_dir / "README.md",
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        run_config=run_config,
        description=args.description,
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "weights": str(weights_path),
                "weights_size_gb": weights_path.stat().st_size / 1024**3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
