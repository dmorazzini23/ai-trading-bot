"""Sleeve horizon parsing and configuration helpers."""
from __future__ import annotations

from typing import Iterable

from ai_trading.config.management import get_env, get_trading_config
from ai_trading.core.netting import SleeveConfig

VALID_TIMEFRAMES = {"1Min", "5Min", "15Min", "1Hour", "1Day"}


def _split_horizons(raw: Iterable[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in raw:
        part = str(item).strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid horizon definition: {item}")
        name, timeframe = [p.strip() for p in part.split(":", 1)]
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        parsed.append((name.lower(), timeframe))
    return parsed


def _cfg_value(cfg: object, name: str, default):
    return getattr(cfg, name, default)


def _resolve_sleeve_symbol_cap(cfg: object, raw_cap: float) -> float:
    """Return sleeve symbol cap aligned to global risk cap when enabled."""

    try:
        align_enabled = bool(
            get_env("AI_TRADING_SLEEVE_SYMBOL_CAP_ALIGN_ENABLED", True, cast=bool)
        )
    except Exception:
        align_enabled = True
    cap = max(0.0, float(raw_cap or 0.0))
    if not align_enabled:
        return cap

    global_cap = float(_cfg_value(cfg, "global_max_symbol_dollars", 0.0) or 0.0)
    if global_cap <= 0.0:
        return cap
    try:
        align_mult = float(
            get_env("AI_TRADING_SLEEVE_SYMBOL_CAP_ALIGN_MULTIPLIER", 1.0, cast=float)
        )
    except Exception:
        align_mult = 1.0
    align_mult = max(0.1, min(align_mult, 5.0))
    effective_global_cap = max(0.0, global_cap * align_mult)
    if effective_global_cap <= 0.0:
        return cap
    if cap <= 0.0:
        return effective_global_cap
    return min(cap, effective_global_cap)


def build_sleeve_configs(cfg=None) -> list[SleeveConfig]:
    cfg = cfg or get_trading_config()
    raw = getattr(cfg, "horizons", ())
    parsed = _split_horizons(raw)
    configs: list[SleeveConfig] = []
    for name, timeframe in parsed:
        if name == "day":
            prefix = "sleeve_day"
        elif name == "swing":
            prefix = "sleeve_swing"
        elif name == "longshort":
            prefix = "sleeve_longshort"
        else:
            raise ValueError(f"Unknown sleeve name: {name}")
        configs.append(
            SleeveConfig(
                name=name,
                timeframe=timeframe,
                enabled=bool(_cfg_value(cfg, f"{prefix}_enabled", True)),
                entry_threshold=float(_cfg_value(cfg, f"{prefix}_entry_threshold", 0.2)),
                exit_threshold=float(_cfg_value(cfg, f"{prefix}_exit_threshold", 0.1)),
                flip_threshold=float(_cfg_value(cfg, f"{prefix}_flip_threshold", 0.3)),
                reentry_threshold=float(_cfg_value(cfg, f"{prefix}_reentry_threshold", 0.6)),
                deadband_dollars=float(_cfg_value(cfg, f"{prefix}_deadband_dollars", 50.0)),
                deadband_shares=float(_cfg_value(cfg, f"{prefix}_deadband_shares", 1.0)),
                turnover_cap_dollars=float(_cfg_value(cfg, f"{prefix}_turnover_cap_dollars", 0.0)),
                cost_k=float(_cfg_value(cfg, f"{prefix}_cost_k", 1.5)),
                edge_scale_bps=float(_cfg_value(cfg, f"{prefix}_edge_scale_bps", 20.0)),
                max_symbol_dollars=_resolve_sleeve_symbol_cap(
                    cfg,
                    float(_cfg_value(cfg, f"{prefix}_max_symbol_dollars", 10000.0)),
                ),
                max_gross_dollars=float(_cfg_value(cfg, f"{prefix}_max_gross_dollars", 50000.0)),
            )
        )
    return configs
