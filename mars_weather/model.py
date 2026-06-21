"""Mars-specific Aurora model construction."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from mars_weather._paths import ensure_aurora_on_path

ensure_aurora_on_path()

from aurora import AuroraPretrained, AuroraSmallPretrained

from mars_weather.openmars import (
    ATMOS_VARS,
    MARS_STATIC_VARS,
    OPENMARS_STEP,
    SURF_VARS,
)


def make_mars_aurora(
    *,
    level_ids: tuple[int, ...],
    size: Literal["base", "small"] = "base",
    load_checkpoint: bool = True,
    timestep: timedelta = OPENMARS_STEP,
    use_lora: bool = False,
    autocast: bool = True,
    model_kwargs: dict[str, Any] | None = None,
):
    """Create an Aurora model configured for OpenMARS variables."""

    kwargs: dict[str, Any] = {
        "surf_vars": SURF_VARS,
        "static_vars": MARS_STATIC_VARS,
        "atmos_vars": ATMOS_VARS,
        "level_condition": level_ids,
        "timestep": timestep,
        "use_lora": use_lora,
        "autocast": autocast,
        "positive_surf_vars": ("ps", "co2ice", "dustcol"),
    }
    if model_kwargs:
        kwargs.update(model_kwargs)

    cls = AuroraSmallPretrained if size == "small" else AuroraPretrained
    model = cls(**kwargs)
    if load_checkpoint:
        model.load_checkpoint(strict=False)
    return model
