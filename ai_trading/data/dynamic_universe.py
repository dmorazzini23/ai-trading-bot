from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from ai_trading.config.management import get_env, get_trading_config
from ai_trading.data.alpaca_screener import (
    MarketMover,
    MarketMoversSnapshot,
    fetch_market_movers,
    fetch_most_actives,
)
from ai_trading.logging import logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.utils.timefmt import utc_now_iso
from ai_trading.utils.universe import normalize_symbol


@dataclass(frozen=True)
class DynamicUniverseConfig:
    enabled: bool = False
    shadow_mode: bool = False
    refresh_sec: int = 300
    gainers_top: int = 10
    losers_top: int = 10
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    min_volume: float = 100_000.0
    prepend: bool = True
    require_etb_shorts: bool = True
    include_most_actives: bool = False
    most_actives_top: int = 10
    most_actives_by: str = "volume"
    snapshot_path: str = "runtime/dynamic_universe_snapshots.jsonl"


@dataclass(frozen=True)
class UniverseCandidate:
    symbol: str
    source: str
    side_bias: str
    rank: int
    pct_change: float
    last_price: float
    volume: float
    dollar_volume: float
    asof: str
    reason: str
    tradable: bool | None = None
    marginable: bool | None = None
    shortable: bool | None = None
    easy_to_borrow: bool | None = None


@dataclass(frozen=True)
class DynamicUniverseResult:
    merged_symbols: list[str]
    additions: list[UniverseCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_float(raw_value: Any, default: float = 0.0) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value


def _extract_value(record: Any, *names: str) -> Any:
    if record is None:
        return None
    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return None


def _bool_from_record(record: Any, *names: str) -> bool | None:
    value = _extract_value(record, *names)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def load_dynamic_universe_config() -> DynamicUniverseConfig:
    return DynamicUniverseConfig(
        enabled=bool(get_env("AI_TRADING_DYNAMIC_UNIVERSE_ENABLED", False, cast=bool)),
        shadow_mode=bool(get_env("AI_TRADING_DYNAMIC_UNIVERSE_SHADOW_MODE", False, cast=bool)),
        refresh_sec=max(0, int(get_env("AI_TRADING_DYNAMIC_UNIVERSE_REFRESH_SEC", 300, cast=int))),
        gainers_top=max(0, int(get_env("AI_TRADING_DYNAMIC_UNIVERSE_GAINERS_TOP", 10, cast=int))),
        losers_top=max(0, int(get_env("AI_TRADING_DYNAMIC_UNIVERSE_LOSERS_TOP", 10, cast=int))),
        min_price=max(0.0, float(get_env("AI_TRADING_DYNAMIC_UNIVERSE_MIN_PRICE", 5.0, cast=float))),
        min_dollar_volume=max(
            0.0,
            float(get_env("AI_TRADING_DYNAMIC_UNIVERSE_MIN_DOLLAR_VOLUME", 5_000_000.0, cast=float)),
        ),
        min_volume=max(0.0, float(get_env("AI_TRADING_DYNAMIC_UNIVERSE_MIN_VOLUME", 100_000.0, cast=float))),
        prepend=bool(get_env("AI_TRADING_DYNAMIC_UNIVERSE_PREPEND", True, cast=bool)),
        require_etb_shorts=bool(
            get_env("AI_TRADING_DYNAMIC_UNIVERSE_REQUIRE_ETB_SHORTS", True, cast=bool)
        ),
        include_most_actives=bool(
            get_env("AI_TRADING_DYNAMIC_UNIVERSE_INCLUDE_MOST_ACTIVES", False, cast=bool)
        ),
        most_actives_top=max(
            0,
            int(get_env("AI_TRADING_DYNAMIC_UNIVERSE_MOST_ACTIVES_TOP", 10, cast=int)),
        ),
        most_actives_by=str(
            get_env("AI_TRADING_DYNAMIC_UNIVERSE_MOST_ACTIVES_BY", "volume", cast=str)
            or "volume"
        ).strip().lower(),
        snapshot_path=str(
            get_env(
                "AI_TRADING_DYNAMIC_UNIVERSE_SNAPSHOT_PATH",
                "runtime/dynamic_universe_snapshots.jsonl",
                cast=str,
            )
            or "runtime/dynamic_universe_snapshots.jsonl"
        ).strip(),
    )


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw_symbol in symbols:
        symbol = normalize_symbol(str(raw_symbol or ""))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _asset_cache(runtime) -> dict[str, Any]:
    existing_cache = getattr(runtime, "_dynamic_universe_asset_cache", None)
    if isinstance(existing_cache, dict):
        return cast(dict[str, Any], existing_cache)
    new_cache: dict[str, Any] = {}
    setattr(runtime, "_dynamic_universe_asset_cache", new_cache)
    return new_cache


def _liquidity_cache(runtime) -> dict[str, dict[str, float]]:
    existing_cache = getattr(runtime, "_dynamic_universe_liquidity_cache", None)
    if isinstance(existing_cache, dict):
        return cast(dict[str, dict[str, float]], existing_cache)
    new_cache: dict[str, dict[str, float]] = {}
    setattr(runtime, "_dynamic_universe_liquidity_cache", new_cache)
    return new_cache


def _resolve_asset(runtime, symbol: str) -> Any:
    cache = _asset_cache(runtime)
    if symbol in cache:
        return cache[symbol]
    candidates = [getattr(runtime, "api", None), getattr(runtime, "trading_client", None)]
    asset = None
    for client in candidates:
        get_asset = getattr(client, "get_asset", None) if client is not None else None
        if not callable(get_asset):
            continue
        try:
            asset = get_asset(symbol)
            break
        except Exception as exc:
            logger.debug(
                "DYNAMIC_UNIVERSE_ASSET_LOOKUP_FAILED",
                extra={"symbol": symbol, "detail": str(exc)},
            )
    cache[symbol] = asset
    return asset


def _resolve_liquidity(runtime, symbol: str, fallback_price: float) -> tuple[float, float, float]:
    cache = _liquidity_cache(runtime)
    cached = cache.get(symbol)
    if isinstance(cached, dict):
        return (
            float(cached.get("price", fallback_price)),
            float(cached.get("volume", 0.0)),
            float(cached.get("dollar_volume", 0.0)),
        )
    data_fetcher = getattr(runtime, "data_fetcher", None)
    get_daily_df = getattr(data_fetcher, "get_daily_df", None) if data_fetcher is not None else None
    last_price = fallback_price
    volume = 0.0
    dollar_volume = 0.0
    if callable(get_daily_df):
        try:
            frame = get_daily_df(runtime, symbol)
        except Exception as exc:
            logger.debug(
                "DYNAMIC_UNIVERSE_LIQUIDITY_FETCH_FAILED",
                extra={"symbol": symbol, "detail": str(exc)},
            )
        else:
            if frame is not None and not getattr(frame, "empty", True):
                try:
                    last_row = frame.iloc[-1]
                    last_price = max(
                        _safe_float(getattr(last_row, "get", lambda *_a: None)("close"), fallback_price),
                        fallback_price,
                    )
                    volume = _safe_float(getattr(last_row, "get", lambda *_a: None)("volume"))
                    dollar_volume = last_price * max(volume, 0.0)
                except Exception as exc:
                    logger.debug(
                        "DYNAMIC_UNIVERSE_LIQUIDITY_PARSE_FAILED",
                        extra={"symbol": symbol, "detail": str(exc)},
                    )
    cache[symbol] = {
        "price": last_price,
        "volume": volume,
        "dollar_volume": dollar_volume,
    }
    return last_price, volume, dollar_volume


def _short_overlay_enabled() -> bool:
    deprecated_flag = get_env("AI_TRADING_ALLOW_SHORT", None, cast=str, resolve_aliases=False)
    if deprecated_flag not in (None, ""):
        raise RuntimeError(
            "AI_TRADING_ALLOW_SHORT is deprecated. Set TRADING__ALLOW_SHORTS instead."
        )
    allow_shorts = bool(get_env("TRADING__ALLOW_SHORTS", True, cast=bool))
    try:
        cfg = get_trading_config()
    except Exception:
        return allow_shorts
    sleeve_enabled = bool(getattr(cfg, "sleeve_longshort_enabled", True))
    return allow_shorts and sleeve_enabled


def _log_candidate_event(event: str, candidate: UniverseCandidate, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "symbol": candidate.symbol,
        "source": candidate.source,
        "side_bias": candidate.side_bias,
        "reason": candidate.reason,
        "price": candidate.last_price,
        "pct_change": candidate.pct_change,
        "volume": candidate.volume,
        "dollar_volume": candidate.dollar_volume,
        "shortable": candidate.shortable,
        "easy_to_borrow": candidate.easy_to_borrow,
        "marginable": candidate.marginable,
        "tradable": candidate.tradable,
    }
    if extra:
        payload.update(extra)
    logger.info(event, extra=payload)


def _candidate_record(candidate: UniverseCandidate) -> dict[str, Any]:
    return asdict(candidate)


def _write_snapshot(config: DynamicUniverseConfig, payload: dict[str, Any]) -> None:
    try:
        target = resolve_runtime_artifact_path(
            config.snapshot_path,
            default_relative="runtime/dynamic_universe_snapshots.jsonl",
            for_write=True,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str))
            handle.write("\n")
    except OSError as exc:
        logger.warning(
            "DYNAMIC_UNIVERSE_SNAPSHOT_WRITE_FAILED",
            extra={"detail": str(exc), "path": config.snapshot_path},
        )


def _build_candidate(
    runtime,
    mover: MarketMover,
    *,
    rank: int,
    source: str,
    side_bias: str,
    asof: str,
    config: DynamicUniverseConfig,
) -> UniverseCandidate | None:
    asset = _resolve_asset(runtime, mover.symbol)
    tradable = _bool_from_record(asset, "tradable", "is_tradable")
    marginable = _bool_from_record(asset, "marginable", "marginable_flag", "is_marginable")
    shortable = _bool_from_record(asset, "shortable", "shortable_flag", "is_shortable")
    easy_to_borrow = _bool_from_record(
        asset,
        "easy_to_borrow",
        "easy_to_borrow_flag",
        "easy_to_borrow_shares",
    )
    price, volume, dollar_volume = _resolve_liquidity(runtime, mover.symbol, mover.price)
    reason = "accepted"
    if tradable is False:
        reason = "asset_not_tradable"
    elif price < config.min_price:
        reason = "price_below_floor"
    elif volume < config.min_volume:
        reason = "volume_below_floor"
    elif dollar_volume < config.min_dollar_volume:
        reason = "dollar_volume_below_floor"
    elif side_bias == "short":
        if shortable is False:
            reason = "asset_not_shortable"
        elif marginable is False:
            reason = "asset_not_marginable"
        elif config.require_etb_shorts and easy_to_borrow is False:
            reason = "asset_not_easy_to_borrow"
    candidate = UniverseCandidate(
        symbol=mover.symbol,
        source=source,
        side_bias=side_bias,
        rank=rank,
        pct_change=mover.percent_change,
        last_price=price,
        volume=volume,
        dollar_volume=dollar_volume,
        asof=asof,
        reason=reason,
        tradable=tradable,
        marginable=marginable,
        shortable=shortable,
        easy_to_borrow=easy_to_borrow,
    )
    if reason != "accepted":
        event_name = (
            "DYNAMIC_UNIVERSE_SHORT_FILTERED"
            if side_bias == "short" and reason.startswith("asset_")
            else "DYNAMIC_UNIVERSE_REJECT"
        )
        _log_candidate_event(event_name, candidate)
        return None
    _log_candidate_event("DYNAMIC_UNIVERSE_ADD", candidate)
    return candidate


def _merge_symbols(base_symbols: list[str], additions: list[UniverseCandidate], *, prepend: bool) -> list[str]:
    dynamic_symbols = [candidate.symbol for candidate in additions]
    if prepend:
        merged = dynamic_symbols + base_symbols
    else:
        merged = base_symbols + dynamic_symbols
    return _dedupe_symbols(merged)


def _metadata_payload(
    base_symbols: list[str],
    merged_symbols: list[str],
    additions: list[UniverseCandidate],
    *,
    config: DynamicUniverseConfig,
    movers_snapshot: MarketMoversSnapshot | None,
    short_overlay_enabled: bool,
    shadow_applied: bool,
) -> dict[str, Any]:
    by_symbol = {candidate.symbol: _candidate_record(candidate) for candidate in additions}
    payload = {
        "enabled": config.enabled,
        "shadow_mode": config.shadow_mode,
        "shadow_applied": shadow_applied,
        "base_count": len(base_symbols),
        "merged_count": len(merged_symbols),
        "dynamic_count": len(additions),
        "short_overlay_enabled": short_overlay_enabled,
        "base_symbols": list(base_symbols),
        "dynamic_symbols": [candidate.symbol for candidate in additions],
        "merged_symbols": list(merged_symbols),
        "candidates": by_symbol,
        "generated_at": utc_now_iso(),
    }
    if movers_snapshot is not None:
        payload["movers"] = {
            "gainers": len(movers_snapshot.gainers),
            "losers": len(movers_snapshot.losers),
            "market_type": movers_snapshot.market_type,
            "last_updated": movers_snapshot.last_updated.isoformat(),
            "used_fallback": movers_snapshot.used_fallback,
        }
    return payload


def build_dynamic_universe(
    runtime,
    base_universe_tickers: list[str],
    *,
    config: DynamicUniverseConfig | None = None,
) -> DynamicUniverseResult:
    cfg = config or load_dynamic_universe_config()
    base_symbols = _dedupe_symbols(list(base_universe_tickers))
    if not cfg.enabled:
        metadata = {
            "enabled": False,
            "base_count": len(base_symbols),
            "merged_count": len(base_symbols),
            "dynamic_count": 0,
            "generated_at": utc_now_iso(),
        }
        return DynamicUniverseResult(
            merged_symbols=base_symbols,
            additions=[],
            metadata=metadata,
        )

    short_overlay = _short_overlay_enabled()
    movers_top = max(cfg.gainers_top, cfg.losers_top, 1)
    movers_snapshot = fetch_market_movers(
        top=movers_top,
        ttl_seconds=cfg.refresh_sec,
    )
    logger.info(
        "DYNAMIC_UNIVERSE_REFRESH",
        extra={
            "base_count": len(base_symbols),
            "gainers_requested": cfg.gainers_top,
            "losers_requested": cfg.losers_top,
            "gainers_returned": len(movers_snapshot.gainers),
            "losers_returned": len(movers_snapshot.losers),
            "used_fallback": movers_snapshot.used_fallback,
            "short_overlay_enabled": short_overlay,
        },
    )

    additions: list[UniverseCandidate] = []
    seen: set[str] = set(base_symbols)
    asof = movers_snapshot.last_updated.isoformat()

    for rank, mover in enumerate(movers_snapshot.gainers[: cfg.gainers_top], start=1):
        candidate = _build_candidate(
            runtime,
            mover,
            rank=rank,
            source="alpaca_gainer",
            side_bias="long",
            asof=asof,
            config=cfg,
        )
        if candidate is None or candidate.symbol in seen:
            continue
        additions.append(candidate)
        seen.add(candidate.symbol)

    if short_overlay:
        for rank, mover in enumerate(movers_snapshot.losers[: cfg.losers_top], start=1):
            candidate = _build_candidate(
                runtime,
                mover,
                rank=rank,
                source="alpaca_loser",
                side_bias="short",
                asof=asof,
                config=cfg,
            )
            if candidate is None or candidate.symbol in seen:
                continue
            additions.append(candidate)
            seen.add(candidate.symbol)
    elif cfg.losers_top > 0 and movers_snapshot.losers:
        logger.info(
            "DYNAMIC_UNIVERSE_SHORT_FILTERED",
            extra={"reason": "short_overlay_disabled", "count": min(len(movers_snapshot.losers), cfg.losers_top)},
        )

    if cfg.include_most_actives and cfg.most_actives_top > 0:
        actives_snapshot = fetch_most_actives(
            top=cfg.most_actives_top,
            by=cfg.most_actives_by,
            ttl_seconds=cfg.refresh_sec,
        )
        logger.info(
            "DYNAMIC_UNIVERSE_MOST_ACTIVES_REFRESH",
            extra={
                "requested": cfg.most_actives_top,
                "returned": len(actives_snapshot.most_actives),
                "used_fallback": actives_snapshot.used_fallback,
                "by": cfg.most_actives_by,
            },
        )
        for rank, active in enumerate(actives_snapshot.most_actives[: cfg.most_actives_top], start=1):
            candidate = _build_candidate(
                runtime,
                MarketMover(
                    symbol=active.symbol,
                    percent_change=0.0,
                    change=0.0,
                    price=0.0,
                ),
                rank=rank,
                source="alpaca_active",
                side_bias="neutral",
                asof=actives_snapshot.last_updated.isoformat(),
                config=cfg,
            )
            if candidate is None or candidate.symbol in seen:
                continue
            additions.append(candidate)
            seen.add(candidate.symbol)

    shadow_applied = bool(cfg.shadow_mode and additions)
    merged_symbols = (
        list(base_symbols)
        if shadow_applied
        else _merge_symbols(base_symbols, additions, prepend=cfg.prepend)
    )
    if not additions and movers_snapshot.used_fallback:
        logger.info(
            "DYNAMIC_UNIVERSE_FALLBACK_STATIC_ONLY",
            extra={"base_count": len(base_symbols)},
        )
    metadata = _metadata_payload(
        base_symbols,
        merged_symbols,
        additions,
        config=cfg,
        movers_snapshot=movers_snapshot,
        short_overlay_enabled=short_overlay,
        shadow_applied=shadow_applied,
    )
    _write_snapshot(cfg, metadata)
    return DynamicUniverseResult(
        merged_symbols=merged_symbols,
        additions=additions,
        metadata=metadata,
    )


__all__ = [
    "DynamicUniverseConfig",
    "DynamicUniverseResult",
    "UniverseCandidate",
    "build_dynamic_universe",
    "load_dynamic_universe_config",
]
