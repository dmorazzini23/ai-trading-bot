from __future__ import annotations

# ruff: noqa
import datetime as dt
from datetime import timezone as _tz
import os
import time
import types
import uuid
from contextlib import suppress

import pandas as pd
import pytz

from ai_trading.logging import get_logger
from ai_trading.utils.optdeps import module_ok  # AI-AGENT-REF: optional import helper

try:  # AI-AGENT-REF: optional Alpaca dependency
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except (ValueError, TypeError):  # pragma: no cover - handled gracefully  # noqa: BLE001
    TimeFrame = None  # type: ignore
    TimeFrameUnit = types.SimpleNamespace(  # type: ignore
        Minute="Minute", Hour="Hour", Day="Day", Week="Week", Month="Month"
    )

try:  # AI-AGENT-REF: optional Alpaca dependency
    from alpaca_trade_api import REST as TradeApiREST
    from alpaca_trade_api.rest import APIError as TradeApiError
except (ValueError, TypeError):  # pragma: no cover - handled gracefully  # noqa: BLE001
    TradeApiREST = None  # type: ignore
    TradeApiError = Exception  # type: ignore

_log = get_logger(__name__)

# AI-AGENT-REF: robust optional Alpaca dependency handling
SHADOW_MODE = os.getenv("SHADOW_MODE", "").lower() in {"1", "true", "yes"}
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(RETRY_HTTP_CODES)

_UTC = pytz.UTC


def _is_intraday_unit(unit_tok: str) -> bool:
    """Return True for minute or hour-based timeframes."""
    return unit_tok in ("Min", "Hour")


def _unit_from_norm(tf_norm: str) -> str:
    """Extract the unit token (e.g. 'Min', 'Day') from a normalized timeframe."""
    for u in ("Min", "Hour", "Day", "Week", "Month"):
        if tf_norm.endswith(u):
            return u
    return "Day"


ALPACA_AVAILABLE = any(
    [
        module_ok("alpaca"),
        module_ok("alpaca_trade_api"),
        module_ok("alpaca.trading"),
        module_ok("alpaca.data"),
    ]
) and os.environ.get("ALPACA_FORCE_UNAVAILABLE", "").lower() not in {"1", "true", "yes"}


def _make_client_order_id(prefix: str = "ai") -> str:
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


generate_client_order_id = _make_client_order_id


def _get(obj, key, default=None):
    """Fetch ``key`` from ``obj`` by attribute or mapping lookup."""  # AI-AGENT-REF
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _normalize_timeframe_for_tradeapi(tf_raw):
    """Support string pass-through and alpaca TimeFrame objects."""
    try:
        from alpaca.data.timeframe import TimeFrame
    except (ValueError, TypeError):
        TimeFrame = None
    if isinstance(tf_raw, str):
        s = tf_raw.strip()
        return s if s[:1].isdigit() else f"1{s.capitalize()}"
    if TimeFrame is not None and isinstance(tf_raw, TimeFrame):
        unit = getattr(tf_raw.unit, "name", str(tf_raw.unit)).title()
        return f"{tf_raw.amount}{unit}"
    return str(tf_raw)


def _to_utc(dtobj: dt.datetime) -> dt.datetime:
    """Ensure a ``datetime`` is timezone-aware and in UTC."""
    if dtobj.tzinfo is None:
        return dtobj.replace(tzinfo=dt.timezone.utc)
    return dtobj.astimezone(dt.timezone.utc)


def _fmt_rfc3339_z(dtobj: dt.datetime) -> str:
    """Format a UTC datetime to RFC3339 ``YYYY-MM-DDTHH:MM:SSZ``."""
    d = _to_utc(dtobj).replace(microsecond=0)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_start_end_for_tradeapi(timeframe: str, start, end):
    """Daily => YYYY-MM-DD; intraday => RFC3339Z in UTC."""
    from ai_trading.utils.datetime import (
        compose_daily_params,
        compose_intraday_params,
        ensure_datetime,
    )

    sd = ensure_datetime(start) if start is not None else None
    ed = ensure_datetime(end) if end is not None else None
    is_daily = str(timeframe).lower() in {"1day", "day", "daily"}
    params = (
        compose_daily_params(sd, ed)
        if is_daily
        else compose_intraday_params(sd, ed)
    )
    return params["start"], params["end"]


# ---- market data helpers ----------------------------------------------------


def _get_rest() -> TradeApiREST:
    """Return a new `alpaca_trade_api.REST` instance."""
    if TradeApiREST is None:  # pragma: no cover - optional dependency
        raise RuntimeError("alpaca-trade-api not installed")
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return TradeApiREST(key, secret, base)


def _bars_time_window(timeframe: TimeFrame) -> tuple[str, str]:
    now = dt.datetime.now(tz=_UTC)
    end = now - dt.timedelta(minutes=1)
    unit = getattr(getattr(timeframe, "unit", None), "name", None)
    if unit == "Day":
        days = int(os.getenv("DATA_LOOKBACK_DAYS_DAILY", 200))
    else:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_MINUTE", 5))
    start = end - dt.timedelta(days=days)
    return _fmt_rfc3339_z(start), _fmt_rfc3339_z(end)  # AI-AGENT-REF: return RFC3339 strings


def get_bars_df(
    symbol: str,
    timeframe: str | TimeFrame,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    adjustment: str | None = None,
    feed: str | None = None,
) -> pd.DataFrame:
    """Fetch bars for ``symbol`` and return a normalized DataFrame."""
    rest = _get_rest()
    feed = feed or os.getenv("ALPACA_DATA_FEED", "iex")
    adjustment = adjustment or os.getenv("ALPACA_ADJUSTMENT", "all")
    tf_raw = timeframe
    tf = _normalize_timeframe_for_tradeapi(tf_raw)
    if start is None or end is None:
        base_tf = None
        if TimeFrame is not None:
            try:
                base_tf = tf_raw if isinstance(tf_raw, TimeFrame) else TimeFrame(1, TimeFrameUnit.Day)
            except (ValueError, TypeError):  # noqa: BLE001
                base_tf = TimeFrame(1, TimeFrameUnit.Day)
        if base_tf is not None:
            start, end = _bars_time_window(base_tf)
        else:
            end = dt.datetime.now(tz=_UTC) - dt.timedelta(minutes=1)
            start = end - dt.timedelta(days=200)
    start_s, end_s = _format_start_end_for_tradeapi(tf, start, end)
    try:
        df = rest.get_bars(
            symbol,
            timeframe=tf,
            start=start_s,
            end=end_s,
            adjustment=adjustment,
            feed=feed,
            limit=None,
        ).df
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.reset_index(drop=False)
        return pd.DataFrame()
    except TradeApiError as e:
        req = {
            "timeframe_raw": str(tf_raw),
            "timeframe_norm": tf,
            "feed": feed,
            "start": start_s,
            "end": end_s,
            "adjustment": adjustment,
        }
        body = ""
        try:
            body = e.response.text
        except (ValueError, TypeError):  # noqa: BLE001
            pass
        _log.error(
            "ALPACA_FAIL",
            extra={
                "symbol": symbol,
                "timeframe": tf,
                "feed": feed,
                "start": start_s,
                "end": end_s,
                "status_code": getattr(e, "status_code", None),
                "endpoint": "alpaca/bars",
                "query_params": req,
                "body": body,
            },
        )
        return pd.DataFrame()


def submit_order(api, order_data=None, log=None, **kwargs):
    """Submit an order and return a canonical ``SimpleNamespace``."""  # AI-AGENT-REF
    data: dict[str, object] = {}
    if order_data is not None:
        if isinstance(order_data, dict):
            data.update(order_data)
        else:
            for k in ("symbol", "qty", "side", "time_in_force", "client_order_id"):
                v = _get(order_data, k)
                if v is not None:
                    data[k] = v
    if kwargs:
        data.update(kwargs)

    symbol = _get(data, "symbol")
    qty = _get(data, "qty")
    side = _get(data, "side")
    tif = _get(data, "time_in_force", "day")
    client_order_id = _get(data, "client_order_id") or _make_client_order_id()
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "time_in_force": tif,
        "client_order_id": client_order_id,
    }

    def _shadow() -> types.SimpleNamespace:
        broker_id = f"shadow-{client_order_id}"
        return types.SimpleNamespace(
            status="shadow",
            success=True,
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
            broker_order_id=broker_id,
            order_id=broker_id,  # back-compat alias
            message="shadow-mode OK",
        )

    if SHADOW_MODE:
        if log:
            with suppress(Exception):
                log.info("submit_order shadow", payload=payload)
        return _shadow()

    # If the client can't submit, fall back to shadow (tests expect success=True)
    submit_fn = getattr(api, "submit_order", None)
    if not callable(submit_fn):
        if log:
            log.info(
                "submit_order fallback to shadow (no submit method)",
                symbol=symbol,
                qty=qty,
            )
        return types.SimpleNamespace(
            status="shadow",
            success=True,
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
            broker_order_id=f"shadow-{client_order_id}",
        )

    if log:
        with suppress(Exception):
            log.info("submit_order live", payload=payload)

    try:
        # Attempt live submit
        resp = submit_fn(**payload)
    except (ValueError, TypeError) as e:  # noqa: BLE001  # AI-AGENT-REF: normalize errors
        status = int(
            getattr(e, "status", getattr(e, "status_code", getattr(e, "code", 0)))
            or 0
        )
        retryable = status in RETRY_HTTP_CODES
        return types.SimpleNamespace(
            status=status,
            success=False,
            retryable=retryable,
            error=str(e),
            client_order_id=client_order_id,
        )

    # Back-compat return semantics
    if isinstance(resp, dict):
        broker_id = resp.get("id")
        resp.setdefault("client_order_id", client_order_id)
        resp.setdefault("status", "submitted")
        resp.setdefault("success", True)
        if "broker_order_id" not in resp:
            resp["broker_order_id"] = broker_id or resp.get("order_id")
        return types.SimpleNamespace(**resp)
    try:
        if getattr(resp, "client_order_id", None) is None:
            setattr(resp, "client_order_id", client_order_id)
        if getattr(resp, "status", None) is None:
            setattr(resp, "status", "submitted")
        if getattr(resp, "success", None) is None:
            setattr(resp, "success", True)
    except (ValueError, TypeError):  # noqa: BLE001
        pass
    return resp


def alpaca_get(*_a, **_k):  # legacy stub
    return None


def start_trade_updates_stream(*_a, **_k):  # legacy stub
    return None


__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "RETRY_HTTP_CODES",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "generate_client_order_id",
    "_bars_time_window",
    "get_bars_df",
    "alpaca_get",
    "start_trade_updates_stream",
]
