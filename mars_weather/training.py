"""Training utilities for OpenMARS fine-tuning."""

from __future__ import annotations

import torch

from mars_weather._paths import ensure_aurora_on_path

ensure_aurora_on_path()

from aurora import Batch

from mars_weather.openmars import ATMOS_VARS, MARS_STATIC_VARS, SURF_VARS


def normalized_mse_loss(
    pred: Batch,
    target: Batch,
    *,
    surf_stats: dict[str, tuple[float, float]] | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute normalized MSE across all configured OpenMARS variables."""

    losses = normalized_mse_losses(pred, target, surf_stats=surf_stats, weights=weights)
    return torch.stack(tuple(losses.values())).mean()


def normalized_mse_losses(
    pred: Batch,
    target: Batch,
    *,
    surf_stats: dict[str, tuple[float, float]] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute normalized MSE per OpenMARS variable."""

    surf_stats = surf_stats or {}
    weights = weights or {}
    pred_n = pred.normalise(surf_stats=surf_stats)
    target_n = target.normalise(surf_stats=surf_stats)

    losses: dict[str, torch.Tensor] = {}
    for name in SURF_VARS:
        losses[name] = weights.get(name, 1.0) * torch.mean(
            (pred_n.surf_vars[name] - target_n.surf_vars[name]) ** 2
        )
    for name in ATMOS_VARS:
        losses[name] = weights.get(name, 1.0) * torch.mean(
            (pred_n.atmos_vars[name] - target_n.atmos_vars[name]) ** 2
        )

    return losses


def make_optimizer(
    model: torch.nn.Module,
    *,
    base_lr: float = 3e-4,
    new_lr: float = 1e-3,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """Create AdamW with higher LR for newly initialized Mars embeddings and heads."""

    mars_names = set(SURF_VARS + ATMOS_VARS + MARS_STATIC_VARS)
    new_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_mars_specific = any(
            f".{var}" in name or name.endswith(var) or f".{var}." in name for var in mars_names
        )
        if is_mars_specific:
            new_params.append(param)
        else:
            base_params.append(param)

    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if new_params:
        groups.append({"params": new_params, "lr": new_lr})

    return torch.optim.AdamW(groups, weight_decay=weight_decay)
