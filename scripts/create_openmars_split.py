#!/usr/bin/env python3
"""Create a reproducible OpenMARS train/validation split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from mars_weather.splits import create_openmars_split_manifest, save_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("splits/openmars_my28-34_train_my35_val.json"),
    )
    parser.add_argument("--train-years", type=int, nargs="+", default=[28, 29, 30, 31, 32, 33, 34])
    parser.add_argument("--val-years", type=int, nargs="+", default=[35])
    parser.add_argument("--history-size", type=int, default=2)
    parser.add_argument("--lead-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(
        path
        for path in args.data_dir.glob("*.nc")
        if path.name != "MCS_ret_coverage.nc"
    )
    manifest = create_openmars_split_manifest(
        files,
        train_years=tuple(args.train_years),
        val_years=tuple(args.val_years),
        root=Path("."),
        history_size=args.history_size,
        lead_steps=args.lead_steps,
    )
    save_split_manifest(manifest, args.output)
    print(f"Wrote {args.output}")
    print(
        "train samples:",
        manifest["splits"]["train"]["num_samples"],
        "val samples:",
        manifest["splits"]["val"]["num_samples"],
    )


if __name__ == "__main__":
    main()
