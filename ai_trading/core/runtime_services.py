"""Runtime health and reconciliation helpers extracted from ``bot_engine.py``."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast


_reconcile_warned = False


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _validate_columns(df: Any, required: Sequence[str], results: dict[str, Any], symbol: str) -> None:
    """Validate that required OHLCV columns are present."""
    be = _bot_engine()
    try:
        df = be.data_fetcher_module.normalize_ohlcv_columns(df)
    except AttributeError:
        pass
    columns = getattr(df, "columns", [])
    missing: list[str] = []
    idx = getattr(df, "index", None)
    has_datetime_index = False
    if idx is not None:
        pd_mod = sys.modules.get("pandas")
        if pd_mod is None:
            try:
                import pandas as pd_mod  # type: ignore
            except ImportError:
                pd_mod = None  # type: ignore[assignment]
        if pd_mod is not None:
            try:
                has_datetime_index = isinstance(idx, pd_mod.DatetimeIndex)
            except AttributeError:
                has_datetime_index = False
        elif getattr(idx, "__class__", None) is not None:
            has_datetime_index = idx.__class__.__name__ == "DatetimeIndex"
    for column in required:
        if column == "timestamp" and has_datetime_index and column not in columns:
            continue
        if column not in columns:
            missing.append(column)
    if missing:
        results["missing_columns"].append(symbol)


def _validate_timezones(df: Any, results: dict[str, Any], symbol: str) -> None:
    """Validate that the dataframe index is timezone-aware when present."""
    if hasattr(df, "index") and hasattr(df.index, "tz") and df.index.tz is None:
        results["timezone_issues"].append(symbol)


def reconcile_positions_runtime(ctx: Any) -> None:
    """Fetch live positions and prune stale stop/take targets."""
    global _reconcile_warned
    be = _bot_engine()
    _reconcile_warned = be._reconcile_position_targets_service(
        ctx,
        logger=be.logger,
        targets_lock=be.targets_lock,
        warned=_reconcile_warned,
    )


def pre_trade_health_check_runtime(
    ctx: Any,
    symbols: Sequence[str],
    min_rows: int | None = None,
) -> dict[str, Any]:
    """Validate symbol data sufficiency and schema sanity for startup/runtime health."""
    be = _bot_engine()

    if min_rows is None:
        min_rows = getattr(ctx, "min_rows", 120)
    min_rows = int(min_rows)

    results: dict[str, Any] = {
        "checked": 0,
        "failures": [],
        "insufficient_rows": [],
        "missing_columns": [],
        "timezone_issues": [],
    }
    if not symbols:
        return results

    settings = be.get_settings()
    now_utc = datetime.now(UTC)
    fallback_days = int(getattr(settings, "pretrade_lookback_days", 120))
    start = getattr(ctx, "lookback_start", now_utc - timedelta(days=fallback_days))
    end = getattr(ctx, "lookback_end", now_utc)
    frames: dict[str, Any] | None = None

    for sym in symbols:
        df = None
        try:
            fetcher = getattr(ctx, "data_fetcher", None)
            if fetcher is not None and hasattr(fetcher, "get_daily_df"):
                df = fetcher.get_daily_df(ctx, sym)
        except (AttributeError, TypeError):
            df = None
        except (ImportError, RuntimeError) as exc:
            message = str(exc).lower()
            if "alpaca stub" in message or "external network blocked in tests" in message:
                df = None
            else:
                raise

        if df is None:
            if frames is None:
                try:
                    frames = be._fetch_universe_bars_chunked(
                        symbols=list(symbols),
                        timeframe="1D",
                        start=start,
                        end=end,
                        feed=getattr(ctx, "data_feed", None),
                    )
                except RuntimeError as exc:
                    if "external network blocked in tests" in str(exc).lower():
                        frames = {}
                    else:
                        raise
            df = (frames or {}).get(sym)

        if df is None or getattr(df, "empty", False):
            results["failures"].append(sym)
            continue

        results["checked"] += 1
        try:
            if len(df) < min_rows:
                results["insufficient_rows"].append(sym)
                continue
            _validate_columns(
                df,
                required=["timestamp", "open", "high", "low", "close", "volume"],
                results=results,
                symbol=sym,
            )
            _validate_timezones(df, results, sym)
        except (
            be.APIError,
            TimeoutError,
            ConnectionError,
            KeyError,
            ValueError,
            TypeError,
            OSError,
        ) as exc:
            results["failures"].append((sym, str(exc)))
            be.logger.warning(
                "HEALTH_CHECK_FAILED",
                extra={"cause": exc.__class__.__name__, "detail": str(exc)},
            )

    return results


def legacy_health_payload_runtime() -> dict[str, Any]:
    """Build the legacy Flask health payload without leaking runtime exceptions."""
    be = _bot_engine()

    runtime = be._get_runtime_context_or_none()
    runtime_not_ready = runtime is None
    runtime_symbols = []
    if runtime is not None:
        tickers = getattr(runtime, "tickers", None)
        if isinstance(tickers, Sequence) and not isinstance(tickers, (str, bytes)):
            runtime_symbols = list(tickers)

    try:
        if runtime is not None:
            pre_trade_health_check_runtime(runtime, runtime_symbols or be.REGIME_SYMBOLS)
        else:
            raise RuntimeError("runtime not ready")
        payload = be.build_canonical_healthz_payload(
            service_name="ai-trading",
            force_ok_for_pytest=False,
            healthy_status_mode="healthy",
            ok_mode="connectivity",
        )
    except (
        be.APIError,
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        AttributeError,
    ) as exc:
        payload = be.build_canonical_healthz_payload(
            service_name="ai-trading",
            force_ok_for_pytest=False,
            healthy_status_mode="healthy",
            ok_mode="connectivity",
            error=str(exc),
        )
        if not runtime_not_ready or str(exc) != "runtime not ready":
            be.logger.warning(
                "HEALTH_CHECK_FAILED",
                extra={"cause": exc.__class__.__name__, "detail": str(exc)},
            )

    payload["no_signal_events"] = int(getattr(be.state, "no_signal_events", 0))
    payload["indicator_failures"] = int(getattr(be.state, "indicator_failures", 0))
    return cast(dict[str, Any], payload)


def start_healthcheck_runtime() -> None:
    """Run the legacy standalone healthcheck Flask app."""
    be = _bot_engine()
    port = be.CFG.healthcheck_port
    try:
        be.app.run(host="0.0.0.0", port=port)
    except OSError as exc:
        be.logger.warning(
            "HEALTHCHECK_PORT_CONFLICT",
            extra={"port": port, "detail": str(exc)},
        )
    except (
        be.APIError,
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as exc:
        be.logger.warning(
            "HEALTH_CHECK_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )


__all__ = [
    "legacy_health_payload_runtime",
    "pre_trade_health_check_runtime",
    "reconcile_positions_runtime",
    "start_healthcheck_runtime",
]
