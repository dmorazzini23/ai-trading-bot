"""Sleeve horizon parsing and configuration helpers."""
from __future__ import annotations

from typing import Iterable

from ai_trading.config.management import get_trading_config
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
                max_symbol_dollars=float(_cfg_value(cfg, f"{prefix}_max_symbol_dollars", 10000.0)),
                max_gross_dollars=float(_cfg_value(cfg, f"{prefix}_max_gross_dollars", 50000.0)),
            )
        )
    return configs
