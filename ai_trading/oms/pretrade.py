"""OMS pre-trade controls for size, collars, duplicates, and throttles."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any

from ai_trading.config.management import get_env


@dataclass(slots=True)
class OrderIntent:
    symbol: str
    side: str
    qty: int
    notional: float
    limit_price: float | None
    bar_ts: datetime
    client_order_id: str
    last_price: float | None = None
    mid: float | None = None
    spread: float | None = None
    sleeve: str | None = None
    liquidity_bucket: str | None = None
    avg_daily_volume: float | None = None
    minute_volume: float | None = None
    expected_slippage_bps: float | None = None
    expected_tca_bps: float | None = None
    fill_quality_score: float | None = None
    quote_quality_ok: bool | None = None
    sector: str | None = None
    factor_name: str | None = None
    factor_exposure: float | None = None


class SlidingWindowRateLimiter:
    """Rate limiter for order and cancel message budgets."""

    def __init__(
        self,
        *,
        global_orders_per_min: int = 0,
        per_symbol_orders_per_min: int = 0,
        cancels_per_min: int = 0,
        cancel_loop_max_without_fill: int = 0,
        cancel_loop_block_bars: int = 0,
    ) -> None:
        self.global_orders_per_min = max(0, int(global_orders_per_min))
        self.per_symbol_orders_per_min = max(0, int(per_symbol_orders_per_min))
        self.cancels_per_min = max(0, int(cancels_per_min))
        self.cancel_loop_max_without_fill = max(0, int(cancel_loop_max_without_fill))
        self.cancel_loop_block_bars = max(0, int(cancel_loop_block_bars))

        self._order_ts: deque[float] = deque()
        self._symbol_order_ts: dict[str, deque[float]] = {}
        self._cancel_ts: deque[float] = deque()
        self._cancel_without_fill: dict[str, int] = {}
        self._symbol_bar_index: dict[str, int] = {}
        self._symbol_last_bar_ts: dict[str, datetime] = {}
        self._symbol_block_until_bar: dict[str, int] = {}

    @staticmethod
    def _now() -> float:
        import time

        return time.monotonic()

    @staticmethod
    def _prune(window: deque[float], now: float, seconds: float) -> None:
        while window and now - window[0] > seconds:
            window.popleft()

    def _advance_bar(self, symbol: str, bar_ts: datetime) -> int:
        ts = bar_ts if bar_ts.tzinfo else bar_ts.replace(tzinfo=UTC)
        previous = self._symbol_last_bar_ts.get(symbol)
        if previous is None or ts > previous:
            self._symbol_last_bar_ts[symbol] = ts
            self._symbol_bar_index[symbol] = self._symbol_bar_index.get(symbol, 0) + 1
        return self._symbol_bar_index.get(symbol, 0)

    def allow_order(self, symbol: str, bar_ts: datetime) -> tuple[bool, str | None, dict[str, Any]]:
        now = self._now()
        bar_idx = self._advance_bar(symbol, bar_ts)
        blocked_until = self._symbol_block_until_bar.get(symbol, 0)
        if blocked_until and bar_idx < blocked_until:
            return False, "CANCEL_LOOP_BLOCK", {"symbol": symbol, "blocked_until_bar": blocked_until}

        self._prune(self._order_ts, now, 60.0)
        if self.global_orders_per_min > 0 and len(self._order_ts) >= self.global_orders_per_min:
            return False, "RATE_THROTTLE_BLOCK", {"scope": "global", "limit": self.global_orders_per_min}

        symbol_window = self._symbol_order_ts.setdefault(symbol, deque())
        self._prune(symbol_window, now, 60.0)
        if (
            self.per_symbol_orders_per_min > 0
            and len(symbol_window) >= self.per_symbol_orders_per_min
        ):
            return False, "RATE_THROTTLE_BLOCK", {"scope": "symbol", "symbol": symbol, "limit": self.per_symbol_orders_per_min}

        return True, None, {}

    def record_order(self, symbol: str, bar_ts: datetime) -> None:
        now = self._now()
        self._advance_bar(symbol, bar_ts)
        self._order_ts.append(now)
        symbol_window = self._symbol_order_ts.setdefault(symbol, deque())
        symbol_window.append(now)

    def record_cancel(self, symbol: str, *, bar_ts: datetime, filled: bool) -> None:
        now = self._now()
        self._cancel_ts.append(now)
        self._prune(self._cancel_ts, now, 60.0)

        if filled:
            self._cancel_without_fill[symbol] = 0
            return

        current = self._cancel_without_fill.get(symbol, 0) + 1
        self._cancel_without_fill[symbol] = current
        if (
            self.cancel_loop_max_without_fill > 0
            and current >= self.cancel_loop_max_without_fill
            and self.cancel_loop_block_bars > 0
        ):
            bar_idx = self._advance_bar(symbol, bar_ts)
            self._symbol_block_until_bar[symbol] = bar_idx + self.cancel_loop_block_bars

    def cancel_rate_ok(self) -> bool:
        if self.cancels_per_min <= 0:
            return True
        now = self._now()
        self._prune(self._cancel_ts, now, 60.0)
        return len(self._cancel_ts) < self.cancels_per_min


def _cfg_value(
    cfg: Any,
    *,
    field: str,
    env_keys: tuple[str, ...],
    default: Any,
    cast: type[float] | type[int],
) -> float | int:
    raw = getattr(cfg, field, None) if cfg is not None else None
    if raw is not None:
        try:
            return cast(raw)
        except (TypeError, ValueError):
            pass
    for env_key in env_keys:
        raw_env = get_env(env_key, None)
        if raw_env is None:
            continue
        return cast(get_env(env_key, default, cast=cast))
    return cast(default)


def _ledger_fingerprints(ledger: Any) -> set[tuple[str, str, int, str]]:
    if ledger is None:
        return set()
    seen = getattr(ledger, "_pretrade_seen_fingerprints", None)
    if isinstance(seen, set):
        return seen
    seen = set()
    setattr(ledger, "_pretrade_seen_fingerprints", seen)
    return seen


def _ledger_position_qty(ledger: Any, symbol: str) -> float | None:
    if ledger is None:
        return None
    symbol_norm = str(symbol).upper()
    method_names = ("position_qty", "get_position_qty", "current_position_qty")
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method(symbol_norm))
            except (TypeError, ValueError):
                continue
    positions = getattr(ledger, "positions", None)
    if isinstance(positions, dict):
        raw_position = positions.get(symbol_norm)
        if raw_position is None:
            return 0.0
        if isinstance(raw_position, (int, float)):
            return float(raw_position)
        raw_qty = getattr(raw_position, "qty", None)
        if raw_qty is not None:
            try:
                return float(raw_qty)
            except (TypeError, ValueError):
                return None
    return None


def _ledger_gross_notional(ledger: Any) -> float | None:
    if ledger is None:
        return None
    method_names = (
        "gross_notional",
        "get_gross_notional",
        "current_gross_notional",
    )
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method())
            except (TypeError, ValueError):
                continue
    attr_names = ("gross_notional", "gross_exposure_notional")
    for attr_name in attr_names:
        value = getattr(ledger, attr_name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _json_env_dict(name: str, default: dict[str, float]) -> dict[str, float]:
    raw = get_env(name, None)
    if raw is None:
        return dict(default)
    parsed: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return dict(default)
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError, json.JSONDecodeError):
            return dict(default)
    if not isinstance(parsed, dict):
        return dict(default)
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key).upper()] = float(value)
        except (TypeError, ValueError):
            continue
    if not out:
        return dict(default)
    return out


def _ledger_sector_notional(ledger: Any, sector: str | None) -> float | None:
    if ledger is None or not sector:
        return None
    method_names = (
        "sector_notional",
        "get_sector_notional",
        "current_sector_notional",
    )
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method(str(sector)))
            except (TypeError, ValueError):
                continue
    sector_map = getattr(ledger, "sector_notional_map", None)
    if isinstance(sector_map, dict):
        try:
            return float(sector_map.get(str(sector), 0.0))
        except (TypeError, ValueError):
            return None
    return None


def _ledger_factor_exposure(ledger: Any, factor_name: str | None) -> float | None:
    if ledger is None or not factor_name:
        return None
    method_names = (
        "factor_exposure",
        "get_factor_exposure",
        "current_factor_exposure",
    )
    for method_name in method_names:
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method(str(factor_name)))
            except (TypeError, ValueError):
                continue
    factor_map = getattr(ledger, "factor_exposure_map", None)
    if isinstance(factor_map, dict):
        try:
            return float(factor_map.get(str(factor_name), 0.0))
        except (TypeError, ValueError):
            return None
    return None


def _ledger_metric_value(ledger: Any, metric: str) -> float | None:
    if ledger is None:
        return None
    metric_key = str(metric).strip().lower()
    metric_method_names: dict[str, tuple[str, ...]] = {
        "var": ("intraday_var", "var_95", "get_intraday_var"),
        "cvar": ("intraday_cvar", "cvar_95", "get_intraday_cvar"),
        "drawdown": ("current_drawdown", "intraday_drawdown", "get_current_drawdown"),
    }
    for method_name in metric_method_names.get(metric_key, ()):
        method = getattr(ledger, method_name, None)
        if callable(method):
            try:
                return float(method())
            except (TypeError, ValueError):
                continue
    attr_names: dict[str, tuple[str, ...]] = {
        "var": ("intraday_var", "var_95"),
        "cvar": ("intraday_cvar", "cvar_95"),
        "drawdown": ("current_drawdown", "intraday_drawdown"),
    }
    for attr_name in attr_names.get(metric_key, ()):
        value = getattr(ledger, attr_name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def validate_pretrade(
    intent: OrderIntent,
    *,
    cfg: Any,
    ledger: Any,
    rate_limiter: SlidingWindowRateLimiter,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate pre-trade controls and return allow/deny decision."""

    max_order_dollars = float(
        _cfg_value(
            cfg,
            field="max_order_dollars",
            env_keys=("MAX_ORDER_DOLLARS", "AI_TRADING_MAX_ORDER_DOLLARS"),
            default=0.0,
            cast=float,
        )
    )
    max_order_shares = int(
        _cfg_value(
            cfg,
            field="max_order_shares",
            env_keys=("MAX_ORDER_SHARES", "AI_TRADING_MAX_ORDER_SHARES"),
            default=0,
            cast=int,
        )
    )
    collar_pct = float(
        _cfg_value(
            cfg,
            field="price_collar_pct",
            env_keys=("PRICE_COLLAR_PCT", "AI_TRADING_PRICE_COLLAR_PCT"),
            default=0.03,
            cast=float,
        )
    )
    max_symbol_notional = float(
        _cfg_value(
            cfg,
            field="max_symbol_notional",
            env_keys=("MAX_SYMBOL_NOTIONAL", "AI_TRADING_MAX_SYMBOL_NOTIONAL"),
            default=0.0,
            cast=float,
        )
    )
    max_gross_notional = float(
        _cfg_value(
            cfg,
            field="max_gross_notional",
            env_keys=("MAX_GROSS_NOTIONAL", "AI_TRADING_MAX_GROSS_NOTIONAL"),
            default=0.0,
            cast=float,
        )
    )
    max_sector_notional = float(
        _cfg_value(
            cfg,
            field="max_sector_notional",
            env_keys=("AI_TRADING_MAX_SECTOR_NOTIONAL",),
            default=0.0,
            cast=float,
        )
    )
    max_factor_exposure = float(
        _cfg_value(
            cfg,
            field="max_factor_exposure",
            env_keys=("AI_TRADING_MAX_FACTOR_EXPOSURE",),
            default=0.0,
            cast=float,
        )
    )
    intraday_var_limit = float(
        _cfg_value(
            cfg,
            field="intraday_var_limit",
            env_keys=("AI_TRADING_INTRADAY_VAR_LIMIT",),
            default=0.0,
            cast=float,
        )
    )
    intraday_cvar_limit = float(
        _cfg_value(
            cfg,
            field="intraday_cvar_limit",
            env_keys=("AI_TRADING_INTRADAY_CVAR_LIMIT",),
            default=0.0,
            cast=float,
        )
    )
    intraday_drawdown_limit = float(
        _cfg_value(
            cfg,
            field="intraday_drawdown_limit",
            env_keys=("AI_TRADING_INTRADAY_DRAWDOWN_LIMIT",),
            default=0.0,
            cast=float,
        )
    )

    qty_abs = abs(int(intent.qty))
    notional_abs = abs(float(intent.notional))
    if (max_order_shares > 0 and qty_abs > max_order_shares) or (
        max_order_dollars > 0 and notional_abs > max_order_dollars
    ):
        return False, "ORDER_SIZE_BLOCK", {"qty": qty_abs, "notional": notional_abs}

    reference = intent.mid if intent.mid and intent.mid > 0 else intent.last_price
    if reference is not None and reference > 0:
        signed_qty = qty_abs if str(intent.side).strip().lower() == "buy" else -qty_abs
        current_symbol_qty = _ledger_position_qty(ledger, intent.symbol)
        if current_symbol_qty is not None:
            projected_symbol_notional = abs((current_symbol_qty + signed_qty) * float(reference))
            if max_symbol_notional > 0 and projected_symbol_notional > max_symbol_notional:
                return (
                    False,
                    "SYMBOL_NOTIONAL_BLOCK",
                    {
                        "symbol": str(intent.symbol).upper(),
                        "projected_symbol_notional": projected_symbol_notional,
                        "max_symbol_notional": max_symbol_notional,
                    },
                )
            current_gross_notional = _ledger_gross_notional(ledger)
            if current_gross_notional is not None and max_gross_notional > 0:
                current_symbol_notional = abs(current_symbol_qty * float(reference))
                projected_gross_notional = max(
                    0.0,
                    float(current_gross_notional) - current_symbol_notional + projected_symbol_notional,
                )
                if projected_gross_notional > max_gross_notional:
                    return (
                        False,
                        "GROSS_NOTIONAL_BLOCK",
                        {
                            "projected_gross_notional": projected_gross_notional,
                            "max_gross_notional": max_gross_notional,
                            "symbol": str(intent.symbol).upper(),
                        },
                    )

    if max_sector_notional > 0 and reference and reference > 0 and intent.sector:
        sector_notional = _ledger_sector_notional(ledger, intent.sector)
        if sector_notional is not None:
            projected_sector_notional = float(sector_notional) + notional_abs
            if projected_sector_notional > max_sector_notional:
                return (
                    False,
                    "SECTOR_CONCENTRATION_BLOCK",
                    {
                        "sector": str(intent.sector),
                        "projected_sector_notional": projected_sector_notional,
                        "max_sector_notional": max_sector_notional,
                    },
                )

    if max_factor_exposure > 0 and intent.factor_name and intent.factor_exposure is not None:
        current_factor = _ledger_factor_exposure(ledger, intent.factor_name)
        projected_factor = float(intent.factor_exposure)
        if current_factor is not None:
            projected_factor += float(current_factor)
        if abs(projected_factor) > max_factor_exposure:
            return (
                False,
                "FACTOR_CONCENTRATION_BLOCK",
                {
                    "factor": str(intent.factor_name),
                    "projected_factor_exposure": projected_factor,
                    "max_factor_exposure": max_factor_exposure,
                },
            )

    if intraday_var_limit > 0:
        current_var = _ledger_metric_value(ledger, "var")
        if current_var is not None and current_var > intraday_var_limit:
            return (
                False,
                "INTRADAY_VAR_BLOCK",
                {"intraday_var": current_var, "max_intraday_var": intraday_var_limit},
            )

    if intraday_cvar_limit > 0:
        current_cvar = _ledger_metric_value(ledger, "cvar")
        if current_cvar is not None and current_cvar > intraday_cvar_limit:
            return (
                False,
                "INTRADAY_CVAR_BLOCK",
                {"intraday_cvar": current_cvar, "max_intraday_cvar": intraday_cvar_limit},
            )

    if intraday_drawdown_limit > 0:
        current_drawdown = _ledger_metric_value(ledger, "drawdown")
        if current_drawdown is not None and current_drawdown > intraday_drawdown_limit:
            return (
                False,
                "INTRADAY_DRAWDOWN_BLOCK",
                {
                    "intraday_drawdown": current_drawdown,
                    "max_intraday_drawdown": intraday_drawdown_limit,
                },
            )

    derisk_enabled = bool(get_env("AI_TRADING_DERISK_ON_DATA_DEGRADED", True, cast=bool))
    derisk_mode = str(get_env("AI_TRADING_DERISK_MODE", "block", cast=str) or "block").strip().lower()
    data_degraded = bool(get_env("AI_TRADING_DATA_DEGRADED", False, cast=bool))
    quote_quality_bad = intent.quote_quality_ok is False
    if derisk_enabled and (data_degraded or quote_quality_bad) and derisk_mode == "block":
        return (
            False,
            "DERISK_DATA_QUALITY_BLOCK",
            {
                "data_degraded": data_degraded,
                "quote_quality_ok": intent.quote_quality_ok,
                "mode": derisk_mode,
            },
        )

    bucket_default = {"THIN": 45.0, "NORMAL": 30.0, "THICK": 20.0}
    bucket_map = _json_env_dict("AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_BUCKET", bucket_default)
    symbol_map = _json_env_dict("AI_TRADING_EXEC_SLIPPAGE_CEILING_BPS_BY_SYMBOL", {})
    liquidity_bucket = str(intent.liquidity_bucket or "NORMAL").upper()
    symbol_ceiling = symbol_map.get(str(intent.symbol).upper())
    bucket_ceiling = bucket_map.get(liquidity_bucket, bucket_map.get("NORMAL", 30.0))
    slippage_ceiling_bps = symbol_ceiling if symbol_ceiling is not None else bucket_ceiling
    if (
        intent.expected_slippage_bps is not None
        and slippage_ceiling_bps is not None
        and float(intent.expected_slippage_bps) > float(slippage_ceiling_bps)
    ):
        return (
            False,
            "SLIPPAGE_CEILING_BLOCK",
            {
                "symbol": str(intent.symbol).upper(),
                "bucket": liquidity_bucket,
                "expected_slippage_bps": float(intent.expected_slippage_bps),
                "ceiling_bps": float(slippage_ceiling_bps),
            },
        )

    max_adv_participation = float(get_env("AI_TRADING_EXEC_MAX_PARTICIPATION_PCT_ADV", 0.05, cast=float))
    max_minute_participation = float(
        get_env("AI_TRADING_EXEC_MAX_PARTICIPATION_PCT_MINUTE", 0.20, cast=float)
    )
    if intent.avg_daily_volume is not None and intent.avg_daily_volume > 0 and max_adv_participation > 0:
        adv_limit = float(intent.avg_daily_volume) * max_adv_participation
        if qty_abs > adv_limit:
            return (
                False,
                "PARTICIPATION_CAP_BLOCK",
                {
                    "scope": "adv",
                    "qty": qty_abs,
                    "adv_limit_qty": adv_limit,
                    "max_participation_pct_adv": max_adv_participation,
                },
            )
    if intent.minute_volume is not None and intent.minute_volume > 0 and max_minute_participation > 0:
        minute_limit = float(intent.minute_volume) * max_minute_participation
        if qty_abs > minute_limit:
            return (
                False,
                "PARTICIPATION_CAP_BLOCK",
                {
                    "scope": "minute",
                    "qty": qty_abs,
                    "minute_limit_qty": minute_limit,
                    "max_participation_pct_minute": max_minute_participation,
                },
            )

    tca_gate_enabled = bool(get_env("AI_TRADING_EXEC_TCA_GATE_ENABLED", True, cast=bool))
    max_expected_tca_bps = float(get_env("AI_TRADING_EXEC_TCA_MAX_EXPECTED_BPS", 35.0, cast=float))
    min_fill_quality = float(get_env("AI_TRADING_EXEC_MIN_FILL_QUALITY_SCORE", 0.50, cast=float))
    if tca_gate_enabled and intent.expected_tca_bps is not None:
        if float(intent.expected_tca_bps) > max_expected_tca_bps:
            return (
                False,
                "TCA_GATE_BLOCK",
                {
                    "expected_tca_bps": float(intent.expected_tca_bps),
                    "max_expected_tca_bps": max_expected_tca_bps,
                },
            )
    if tca_gate_enabled and intent.fill_quality_score is not None:
        if float(intent.fill_quality_score) < min_fill_quality:
            return (
                False,
                "FILL_QUALITY_GATE_BLOCK",
                {
                    "fill_quality_score": float(intent.fill_quality_score),
                    "min_fill_quality_score": min_fill_quality,
                },
            )

    if intent.limit_price is not None and reference and reference > 0:
        deviation = abs(float(intent.limit_price) - float(reference)) / float(reference)
        if deviation > max(0.0, collar_pct):
            return False, "PRICE_COLLAR_BLOCK", {"reference": reference, "limit_price": intent.limit_price, "deviation": deviation}

    if ledger is not None and intent.client_order_id:
        seen_fn = getattr(ledger, "seen_client_order_id", None)
        if callable(seen_fn) and bool(seen_fn(intent.client_order_id)):
            return False, "DUPLICATE_ORDER_BLOCK", {"client_order_id": intent.client_order_id}

    fingerprint = (
        str(intent.symbol).upper(),
        str(intent.side).lower(),
        qty_abs,
        intent.bar_ts.isoformat(),
    )
    fingerprints = _ledger_fingerprints(ledger)
    if fingerprint in fingerprints:
        return False, "DUPLICATE_ORDER_BLOCK", {"fingerprint": fingerprint}

    allowed, reason, details = rate_limiter.allow_order(intent.symbol, intent.bar_ts)
    if not allowed:
        return False, reason or "RATE_THROTTLE_BLOCK", details
    if not rate_limiter.cancel_rate_ok():
        return False, "RATE_THROTTLE_BLOCK", {"scope": "cancel"}

    rate_limiter.record_order(intent.symbol, intent.bar_ts)
    if ledger is not None:
        fingerprints.add(fingerprint)
    return True, "OK", {}


def safe_validate_pretrade(
    intent: OrderIntent,
    *,
    cfg: Any,
    ledger: Any,
    rate_limiter: SlidingWindowRateLimiter,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate pre-trade controls with fail-closed handling on unexpected errors."""

    fail_closed = bool(get_env("AI_TRADING_PRETRADE_FAIL_CLOSED", True, cast=bool))
    try:
        return validate_pretrade(
            intent,
            cfg=cfg,
            ledger=ledger,
            rate_limiter=rate_limiter,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        details = {
            "error": str(exc),
            "symbol": str(intent.symbol).upper(),
            "client_order_id": intent.client_order_id,
            "fail_closed": fail_closed,
        }
        if fail_closed:
            return False, "PRETRADE_VALIDATION_ERROR", details
        return True, "PRETRADE_VALIDATION_FAIL_OPEN", details
