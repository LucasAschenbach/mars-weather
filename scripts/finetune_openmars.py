#!/usr/bin/env python3
"""Fine-tune Aurora on OpenMARS NetCDF files."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mars_weather import (
    OpenMARSDataset,
    collate_batch_pairs,
    compute_openmars_stats,
    dataset_from_manifest,
    load_openmars_stats,
    load_split_manifest,
    make_mars_aurora,
    make_optimizer,
    normalized_mse_loss,
    normalized_mse_losses,
    register_openmars_stats,
    save_openmars_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="OpenMARS NetCDF files.")
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--stats", type=Path, default=Path("artifacts/openmars_stats.json"))
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--stats-max-time-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--dataset-cache-size", type=int, default=8)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-size", choices=("base", "small"), default="base")
    parser.add_argument("--no-load-checkpoint", action="store_true")
    parser.add_argument("--no-aurora-autocast", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--distributed", choices=("none", "fsdp"), default="none")
    parser.add_argument("--fsdp-min-params", type=int, default=10_000_000)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--new-lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/openmars_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=50)
    parser.add_argument("--full-val-every-epoch", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def init_distributed(mode: str) -> tuple[bool, int, int, int]:
    if mode == "none":
        return False, 0, 0, 1

    if mode != "fsdp":
        raise ValueError(f"Unsupported distributed mode: {mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP training requires CUDA.")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, local_rank, rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def resolve_run_dirs(args: argparse.Namespace, *, distributed: bool) -> tuple[Path, Path]:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = args.run_name or f"{args.split}_{args.distributed}_{args.model_size}"
        output_dir = args.runs_dir / f"{timestamp}_{suffix}"

    tensorboard_dir = args.tensorboard_dir or (output_dir / "tensorboard")
    if distributed:
        object_list: list[str | None] = [None, None]
        if dist.get_rank() == 0:
            object_list = [str(output_dir), str(tensorboard_dir)]
        dist.broadcast_object_list(object_list, src=0)
        output_dir = Path(str(object_list[0]))
        tensorboard_dir = Path(str(object_list[1]))

    return output_dir, tensorboard_dir


def write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["output_dir_resolved"] = str(output_dir)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _run_command(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        },
        "git": {
            "commit": _run_command(["git", "rev-parse", "HEAD"]),
            "status_short": _run_command(["git", "status", "--short"]),
        },
        "files": {},
    }
    if args.split_manifest and args.split_manifest.exists():
        metadata["files"]["split_manifest"] = {
            "path": str(args.split_manifest),
            "sha256": file_sha256(args.split_manifest),
        }
    if args.stats.exists():
        metadata["files"]["stats"] = {
            "path": str(args.stats),
            "sha256": file_sha256(args.stats),
        }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def reduce_mean(value: torch.Tensor, *, distributed: bool, world_size: int) -> torch.Tensor:
    if not distributed:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def reduce_sum(value: torch.Tensor, *, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    distributed: bool,
    rank: int,
    max_batches: int | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    variable_sums: dict[str, torch.Tensor] = {}
    count = torch.zeros((), device=device)

    with torch.inference_mode():
        for batch_i, (batch, target) in enumerate(loader):
            if max_batches is not None and batch_i >= max_batches:
                break
            batch = batch.to(device)
            target = target.to(device)
            pred = model(batch)
            losses = normalized_mse_losses(pred, target)
            batch_size = next(iter(target.surf_vars.values())).shape[0]
            for name, loss in losses.items():
                if name not in variable_sums:
                    variable_sums[name] = torch.zeros((), device=device)
                variable_sums[name] += loss.detach() * batch_size
            count += batch_size

    count = reduce_sum(count, distributed=distributed)
    reduced_losses = {
        name: reduce_sum(value, distributed=distributed) / count.clamp_min(1)
        for name, value in variable_sums.items()
    }
    if was_training:
        model.train()

    if is_rank_zero(rank):
        losses = {name: float(value.cpu()) for name, value in reduced_losses.items()}
        losses["loss"] = sum(losses.values()) / max(len(reduced_losses), 1)
        return losses
    return {}


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    stats,
    distributed: bool,
    rank: int,
) -> None:
    checkpoint = None
    if distributed:
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_config,
            optim_config,
        ):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
        if is_rank_zero(rank):
            checkpoint = {
                "model": model_state,
                "optimizer": optimizer_state,
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
                "stats": stats.to_json_dict(),
                "distributed": "fsdp",
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            }
    elif is_rank_zero(rank):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "stats": stats.to_json_dict(),
            "distributed": "none",
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    if checkpoint is not None:
        torch.save(checkpoint, output_path)


def update_latest_checkpoint(output_dir: Path, checkpoint_path: Path) -> None:
    latest = output_dir / "checkpoint_latest.pt"
    tmp = output_dir / ".checkpoint_latest.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(checkpoint_path.name)
    os.replace(tmp, latest)


def prune_checkpoints(output_dir: Path, *, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("checkpoint_step_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    for path in checkpoints[:-keep_last]:
        path.unlink(missing_ok=True)


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    distributed: bool,
    device: str | torch.device,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if distributed:
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_config,
            optim_config,
        ):
            model.load_state_dict(checkpoint["model"])
            optimizer_state = FSDP.optim_state_dict_to_load(
                model,
                optimizer,
                checkpoint["optimizer"],
            )
            optimizer.load_state_dict(optimizer_state)
    else:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    return (
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("step_in_epoch", -1)),
        int(checkpoint.get("global_step", 0)),
    )


def set_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    new_lr: float,
) -> None:
    """Apply CLI learning rates after loading a checkpointed optimizer state."""

    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]["lr"] = base_lr
        return

    optimizer.param_groups[0]["lr"] = base_lr
    optimizer.param_groups[1]["lr"] = new_lr
    for group in optimizer.param_groups[2:]:
        group["lr"] = new_lr


def main() -> None:
    args = parse_args()
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    distributed, local_rank, rank, world_size = init_distributed(args.distributed)
    output_dir, tensorboard_dir = resolve_run_dirs(args, distributed=distributed)
    if is_rank_zero(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_config(args, output_dir)
    writer = None
    if is_rank_zero(rank) and not args.no_tensorboard:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"

    if args.split_manifest:
        manifest = load_split_manifest(args.split_manifest)
        dataset = dataset_from_manifest(
            manifest,
            args.split,
            dataset_cache_size=args.dataset_cache_size,
        )
        val_dataset = (
            dataset_from_manifest(
                manifest,
                "val",
                dataset_cache_size=args.dataset_cache_size,
            )
            if args.val_every > 0 and args.split == "train"
            else None
        )
        stats_files = [Path(file_info["path"]) for file_info in manifest["files"]]
        stats_years = tuple(int(year) for year in manifest["splits"]["train"]["target_mars_years"])
    else:
        if not args.files:
            raise SystemExit("Provide OpenMARS files or --split-manifest.")
        dataset = OpenMARSDataset(args.files, dataset_cache_size=args.dataset_cache_size)
        val_dataset = None
        stats_files = args.files
        stats_years = None

    if args.recompute_stats or not args.stats.exists():
        if is_rank_zero(rank):
            stats = compute_openmars_stats(
                stats_files,
                max_time_steps=args.stats_max_time_steps,
                mars_years=stats_years,
            )
            save_openmars_stats(stats, args.stats)
        if distributed:
            dist.barrier()
    stats = load_openmars_stats(args.stats, register=False)
    register_openmars_stats(stats)
    if is_rank_zero(rank):
        write_run_metadata(args, output_dir)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collate_batch_pairs,
        "persistent_workers": args.num_workers > 0 and not args.no_persistent_workers,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **dataloader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if distributed
            else None
        )
        val_kwargs = dict(dataloader_kwargs)
        val_kwargs.update(
            {
                "shuffle": False,
                "sampler": val_sampler,
            }
        )
        val_loader = DataLoader(val_dataset, **val_kwargs)

    model = make_mars_aurora(
        level_ids=stats.level_ids,
        size=args.model_size,
        load_checkpoint=not args.no_load_checkpoint,
        autocast=not args.no_aurora_autocast,
    )
    if not args.no_activation_checkpointing:
        model.configure_activation_checkpointing()
    model.train()
    if distributed:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=args.fsdp_min_params,
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
    else:
        model.to(device)

    optimizer = make_optimizer(model, base_lr=args.base_lr, new_lr=args.new_lr)

    start_epoch = 0
    resume_step_in_epoch = -1
    global_step = 0
    if args.resume is not None:
        start_epoch, resume_step_in_epoch, global_step = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            distributed=distributed,
            device=device,
        )
        set_optimizer_learning_rates(
            optimizer,
            base_lr=args.base_lr,
            new_lr=args.new_lr,
        )
        if distributed:
            dist.barrier()

    last_log_time = time.perf_counter()
    last_log_step = global_step
    try:
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            progress = tqdm(
                loader,
                desc=f"epoch {epoch + 1}/{args.epochs}",
                disable=not is_rank_zero(rank),
            )
            stop_after_epoch = False
            last_step_in_epoch = -1
            batch_end_time = time.perf_counter()
            for step_in_epoch, (batch, target) in enumerate(progress):
                if epoch == start_epoch and step_in_epoch <= resume_step_in_epoch:
                    continue
                last_step_in_epoch = step_in_epoch
                step_start_time = time.perf_counter()
                data_wait_time = step_start_time - batch_end_time
                batch = batch.to(device)
                target = target.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
                train_losses = normalized_mse_losses(pred, target)
                loss = torch.stack(tuple(train_losses.values())).mean()
                loss.backward()
                if args.grad_clip > 0:
                    if distributed:
                        model.clip_grad_norm_(args.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step_end_time = time.perf_counter()
                step_time = step_end_time - step_start_time

                global_step += 1
                should_log = global_step % args.log_every == 0
                logged_loss = reduce_mean(loss, distributed=distributed, world_size=world_size)
                logged_train_losses = {
                    name: reduce_mean(value, distributed=distributed, world_size=world_size)
                    for name, value in train_losses.items()
                } if should_log else {}
                if is_rank_zero(rank):
                    loss_value = float(logged_loss.cpu())
                    progress.set_postfix(loss=f"{loss_value:.5f}")
                    if writer is not None and should_log:
                        now = time.perf_counter()
                        elapsed = max(now - last_log_time, 1e-9)
                        step_delta = max(global_step - last_log_step, 1)
                        samples_delta = step_delta * args.batch_size * world_size
                        writer.add_scalar("train/loss", loss_value, global_step)
                        writer.add_scalar("train/epoch", epoch + 1, global_step)
                        writer.add_scalar("train/samples", global_step * args.batch_size * world_size, global_step)
                        writer.add_scalar("train/steps_per_sec", step_delta / elapsed, global_step)
                        writer.add_scalar("train/samples_per_sec", samples_delta / elapsed, global_step)
                        writer.add_scalar("time/data_wait_sec", data_wait_time, global_step)
                        writer.add_scalar("time/step_sec", step_time, global_step)
                        writer.add_scalar(
                            "time/compute_sec",
                            step_time,
                            global_step,
                        )
                        for name, variable_loss in logged_train_losses.items():
                            writer.add_scalar(
                                f"train/loss/{name}",
                                float(variable_loss.cpu()),
                                global_step,
                            )
                        for group_i, group in enumerate(optimizer.param_groups):
                            writer.add_scalar(f"optimizer/lr_group_{group_i}", group["lr"], global_step)
                        if torch.cuda.is_available():
                            device_index = (
                                local_rank if distributed else torch.cuda.current_device()
                            )
                            writer.add_scalar(
                                "gpu/memory_allocated_gb",
                                torch.cuda.memory_allocated(device_index) / 1024**3,
                                global_step,
                            )
                            writer.add_scalar(
                                "gpu/max_memory_allocated_gb",
                                torch.cuda.max_memory_allocated(device_index) / 1024**3,
                                global_step,
                            )
                            writer.add_scalar(
                                "gpu/memory_reserved_gb",
                                torch.cuda.memory_reserved(device_index) / 1024**3,
                                global_step,
                            )
                        last_log_time = now
                        last_log_step = global_step
                if val_loader is not None and global_step % args.val_every == 0:
                    val_losses = evaluate(
                        model=model,
                        loader=val_loader,
                        device=device,
                        distributed=distributed,
                        rank=rank,
                        max_batches=args.val_max_batches,
                    )
                    if is_rank_zero(rank):
                        progress.set_postfix(
                            loss=f"{loss_value:.5f}",
                            val_loss=f"{val_losses['loss']:.5f}",
                        )
                        if writer is not None:
                            writer.add_scalar("val/loss", val_losses["loss"], global_step)
                            for name, value in val_losses.items():
                                if name != "loss":
                                    writer.add_scalar(f"val/loss/{name}", value, global_step)
                if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                    checkpoint_path = output_dir / f"checkpoint_step_{global_step}.pt"
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        output_path=checkpoint_path,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        stats=stats,
                        distributed=distributed,
                        rank=rank,
                    )
                    if is_rank_zero(rank):
                        update_latest_checkpoint(output_dir, checkpoint_path)
                        prune_checkpoints(output_dir, keep_last=args.keep_last)
                    if distributed:
                        dist.barrier()
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_after_epoch = True
                    break
                batch_end_time = time.perf_counter()

            if (
                val_loader is not None
                and args.full_val_every_epoch
                and not stop_after_epoch
            ):
                val_losses = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    distributed=distributed,
                    rank=rank,
                    max_batches=None,
                )
                if is_rank_zero(rank) and writer is not None:
                    writer.add_scalar("val_epoch/loss", val_losses["loss"], global_step)
                    for name, value in val_losses.items():
                        if name != "loss":
                            writer.add_scalar(f"val_epoch/loss/{name}", value, global_step)

            checkpoint_path = output_dir / f"checkpoint_step_{global_step}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_path=checkpoint_path,
                epoch=epoch,
                step_in_epoch=last_step_in_epoch,
                global_step=global_step,
                stats=stats,
                distributed=distributed,
                rank=rank,
            )
            if is_rank_zero(rank):
                update_latest_checkpoint(output_dir, checkpoint_path)
                prune_checkpoints(output_dir, keep_last=args.keep_last)
            if distributed:
                dist.barrier()
            if stop_after_epoch:
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
