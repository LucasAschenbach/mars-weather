#!/usr/bin/env python3
"""Download OpenMARS NetCDF files used by this project."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm


DEFAULT_BASE_URL = "https://ndownloader.figshare.com/files"
DEFAULT_MANIFEST = Path("splits/openmars_my28-34_train_my35_val.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Split manifest whose file list determines what to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where NetCDF files will be written.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional explicit Figshare file IDs. Overrides --manifest.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Open each downloaded NetCDF after transfer to catch corrupt files.",
    )
    return parser.parse_args()


def file_ids_from_manifest(path: Path) -> list[str]:
    manifest = json.loads(path.read_text())
    ids = []
    for file_info in manifest["files"]:
        file_id = Path(file_info["path"]).stem
        if not file_id.isdigit():
            raise ValueError(f"Cannot infer Figshare file ID from {file_info['path']!r}.")
        ids.append(file_id)
    return ids


def remote_size(url: str, *, timeout: float) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
    except (urllib.error.URLError, TimeoutError):
        return None
    return int(value) if value and value.isdigit() else None


def validate_netcdf(path: Path) -> None:
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4") as ds:
        required = {"ps", "tsurf", "co2ice", "dustcol", "u", "v", "temp", "lat", "lon", "lev"}
        missing = sorted(required.difference(ds.variables))
        if missing:
            raise ValueError(f"{path} is missing variables: {', '.join(missing)}")


def download_one(
    file_id: str,
    *,
    output_dir: Path,
    base_url: str,
    timeout: float,
    retries: int,
    overwrite: bool,
    validate: bool,
) -> dict:
    output_path = output_dir / f"{file_id}.nc"
    part_path = output_dir / f"{file_id}.nc.part"
    url = f"{base_url.rstrip('/')}/{file_id}"

    expected_size = remote_size(url, timeout=timeout)
    if output_path.exists() and not overwrite:
        if expected_size is None or output_path.stat().st_size == expected_size:
            if validate:
                validate_netcdf(output_path)
            return {"file_id": file_id, "status": "exists", "path": str(output_path)}

    if overwrite:
        output_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resume_at = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={resume_at}-"} if resume_at > 0 else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if resume_at > 0 and response.status != 206:
                    resume_at = 0
                    part_path.unlink(missing_ok=True)
                mode = "ab" if resume_at > 0 else "wb"
                with part_path.open(mode) as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            if expected_size is not None and part_path.stat().st_size != expected_size:
                raise IOError(
                    f"size mismatch for {file_id}: got {part_path.stat().st_size}, "
                    f"expected {expected_size}"
                )
            part_path.replace(output_path)
            if validate:
                validate_netcdf(output_path)
            return {"file_id": file_id, "status": "downloaded", "path": str(output_path)}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))

    return {"file_id": file_id, "status": "failed", "error": str(last_error)}


def main() -> None:
    args = parse_args()
    file_ids = args.ids if args.ids is not None else file_ids_from_manifest(args.manifest)
    file_ids = sorted(dict.fromkeys(str(file_id) for file_id in file_ids))

    if args.dry_run:
        for file_id in file_ids:
            print(f"{args.base_url.rstrip('/')}/{file_id} -> {args.output_dir / (file_id + '.nc')}")
        return

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = [
            pool.submit(
                download_one,
                file_id,
                output_dir=args.output_dir,
                base_url=args.base_url,
                timeout=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
                validate=args.validate,
            )
            for file_id in file_ids
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="OpenMARS files",
        ):
            results.append(future.result())

    failed = [result for result in results if result["status"] == "failed"]
    downloaded = sum(1 for result in results if result["status"] == "downloaded")
    existing = sum(1 for result in results if result["status"] == "exists")
    print(
        json.dumps(
            {
                "files": len(results),
                "downloaded": downloaded,
                "existing": existing,
                "failed": failed,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
