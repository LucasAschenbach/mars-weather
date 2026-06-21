#!/usr/bin/env python3
"""Export an OpenMARS/Aurora NetCDF forecast to app-readable static frames."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


SURFACE_ORDER = ("ps", "tsurf", "co2ice", "dustcol")
ATMOS_ORDER = ("u", "v", "temp")
STEP_HOURS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OpenMARS/Aurora NetCDF file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/forecasts/latest"),
        help="Directory to write manifest.json and frames/*.f32.",
    )
    parser.add_argument(
        "--max-lead-hours",
        type=int,
        default=168,
        help="Maximum lead time to export. Use -1 to export every frame.",
    )
    parser.add_argument(
        "--source",
        default="aurora-openmars",
        help="Source label written into manifest.json.",
    )
    return parser.parse_args()


def open_dataset(path: Path) -> xr.Dataset:
    for engine in ("h5netcdf", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception:
            continue
    return xr.open_dataset(path)


def variable_ranges(ds: xr.Dataset, frame_count: int) -> dict[str, tuple[float, float]]:
    ranges = {}
    for name in SURFACE_ORDER + ATMOS_ORDER:
        values = np.asarray(ds[name].isel(time=slice(0, frame_count)).values)
        ranges[name] = (float(np.nanmin(values)), float(np.nanmax(values)))
    return ranges


def encode_int16(values, value_min: float, value_max: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if value_max <= value_min:
        return np.zeros(values.shape, dtype="<i2")
    scale = (value_max - value_min) / 65534.0
    encoded = np.rint((values - value_min) / scale - 32767.0)
    return np.clip(encoded, -32767, 32767).astype("<i2")


def variable_meta(unit: str, dims: list[str], value_min: float, value_max: float) -> dict:
    scale = (value_max - value_min) / 65534.0 if value_max > value_min else 1.0
    return {
        "unit": unit,
        "dims": dims,
        "encoding": {
            "dtype": "int16",
            "min": value_min,
            "scale": scale,
        },
    }


def main() -> None:
    args = parse_args()
    ds = open_dataset(args.input)

    missing = [name for name in SURFACE_ORDER + ATMOS_ORDER if name not in ds]
    if missing:
        raise SystemExit(f"Missing required variables: {', '.join(missing)}")

    output = args.output
    frames_dir = output / "frames"
    if output.exists():
        shutil.rmtree(output)
    frames_dir.mkdir(parents=True)

    num_times = int(ds.sizes["time"])
    if args.max_lead_hours < 0:
        frame_count = num_times
    else:
        frame_count = min(num_times, args.max_lead_hours // STEP_HOURS + 1)

    ranges = variable_ranges(ds, frame_count)
    frames = []
    for time_i in range(frame_count):
        lead_hours = time_i * STEP_HOURS
        frame_name = f"lead-{lead_hours:03d}.i16"
        arrays = []
        arrays.extend(
            encode_int16(ds[name].isel(time=time_i).values, *ranges[name]).ravel()
            for name in SURFACE_ORDER
        )
        arrays.extend(
            encode_int16(ds[name].isel(time=time_i).values, *ranges[name]).ravel()
            for name in ATMOS_ORDER
        )
        np.concatenate(arrays).tofile(frames_dir / frame_name)
        frames.append({"leadHours": lead_hours, "path": f"frames/{frame_name}"})

    manifest = {
        "schema": "mars-weather-forecast-v1",
        "source": args.source,
        "stepHours": STEP_HOURS,
        "grid": {
            "lat": [float(v) for v in ds.lat.values],
            "lon": [float(v) for v in ds.lon.values],
            "lev": [float(v) for v in ds.lev.values],
        },
        "variables": {
            "ps": variable_meta("Pa", ["lat", "lon"], *ranges["ps"]),
            "tsurf": variable_meta("K", ["lat", "lon"], *ranges["tsurf"]),
            "co2ice": variable_meta("kg/m2", ["lat", "lon"], *ranges["co2ice"]),
            "dustcol": variable_meta("opacity", ["lat", "lon"], *ranges["dustcol"]),
            "u": variable_meta("m/s", ["lev", "lat", "lon"], *ranges["u"]),
            "v": variable_meta("m/s", ["lev", "lat", "lon"], *ranges["v"]),
            "temp": variable_meta("K", ["lev", "lat", "lon"], *ranges["temp"]),
        },
        "frames": frames,
    }
    if "Ls" in ds:
        manifest["solarLongitude"] = [float(v) for v in ds.Ls.isel(time=slice(0, frame_count)).values]
    if "MY" in ds:
        manifest["marsYear"] = [int(v) for v in ds.MY.isel(time=slice(0, frame_count)).values]

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": frame_count}, indent=2))


if __name__ == "__main__":
    main()
