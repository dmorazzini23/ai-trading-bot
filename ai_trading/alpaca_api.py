from __future__ import annotations
import datetime as dt
from datetime import timezone
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import requests
import importlib.util
from ai_trading.logging import get_logger
from ai_trading.config.management import is_shadow_mode

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_log = get_logger(__name__)
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(RETRY_HTTP_CODES)
_UTC = timezone.utc  # AI-AGENT-REF: prefer stdlib UTC


from zoneinfo import ZoneInfo


def eastern_tz() -> ZoneInfo:
    """Return America/New_York tzinfo using stdlib zoneinfo (Py3.12)."""
    return ZoneInfo("America/New_York")  # AI-AGENT-REF: rely solely on stdlib


EASTERN_TZ = eastern_tz()

def _is_intraday_unit(unit_tok: str) -> bool:
    """Return True for minute or hour-based timeframes."""
    return unit_tok in ('Min', 'Hour')

def _unit_from_norm(tf_norm: str) -> str:
    """Extract the unit token (e.g. 'Min', 'Day') from a normalized timeframe."""
    for u in ('Min', 'Hour', 'Day', 'Week', 'Month'):
        if tf_norm.endswith(u):
            return u
    return 'Day'

def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


HAS_PANDAS: bool = _module_exists("pandas")  # AI-AGENT-REF: expose pandas availability

def _make_client_order_id(prefix: str='ai') -> str:
    return f'{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}'
generate_client_order_id = _make_client_order_id

def _normalize_timeframe_for_tradeapi(tf_raw):
    """Support string pass-through and alpaca TimeFrame objects."""
    try:
        from alpaca.data.timeframe import TimeFrame
    except (ValueError, TypeError, ImportError):
        TimeFrame = None
    if isinstance(tf_raw, str):
        s = tf_raw.strip()
        return s if s[:1].isdigit() else f'1{s.capitalize()}'
    if TimeFrame is not None and isinstance(tf_raw, TimeFrame):
        unit = getattr(tf_raw.unit, 'name', str(tf_raw.unit)).title()
        return f'{tf_raw.amount}{unit}'
    return str(tf_raw)

def _to_utc(dtobj: dt.datetime) -> dt.datetime:
    """Ensure a ``datetime`` is timezone-aware and in UTC."""
    if dtobj.tzinfo is None:
        return dtobj.replace(tzinfo=dt.timezone.utc)
    return dtobj.astimezone(dt.timezone.utc)

def _fmt_rfc3339_z(dtobj: dt.datetime) -> str:
    """Format a UTC datetime to RFC3339 ``YYYY-MM-DDTHH:MM:SSZ``."""
    d = _to_utc(dtobj).replace(microsecond=0)
    return d.strftime('%Y-%m-%dT%H:%M:%SZ')

def _format_start_end_for_tradeapi(timeframe: str, start, end):
    """Daily => YYYY-MM-DD; intraday => RFC3339Z in UTC."""
    from ai_trading.utils.datetime import compose_daily_params, compose_intraday_params, ensure_datetime
    sd = ensure_datetime(start) if start is not None else None
    ed = ensure_datetime(end) if end is not None else None
    is_daily = str(timeframe).lower() in {'1day', 'day', 'daily'}
    params = compose_daily_params(sd, ed) if is_daily else compose_intraday_params(sd, ed)
    return (params['start'], params['end'])

def _get_rest() -> Any:
    """Return a new `alpaca.trading.client.TradingClient` instance."""
    from alpaca.trading.client import TradingClient

    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET_KEY')
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    return TradingClient(key, secret, base_url=base)

def _bars_time_window(timeframe: Any) -> tuple[str, str]:
    now = dt.datetime.now(tz=_UTC)
    end = now - dt.timedelta(minutes=1)
    unit = getattr(getattr(timeframe, 'unit', None), 'name', None)
    if unit == 'Day':
        days = int(os.getenv('DATA_LOOKBACK_DAYS_DAILY', 200))
    else:
        days = int(os.getenv('DATA_LOOKBACK_DAYS_MINUTE', 5))
    start = end - dt.timedelta(days=days)
    return (_fmt_rfc3339_z(start), _fmt_rfc3339_z(end))


# AI-AGENT-REF: ensure clear error when pandas missing
def _require_pandas(consumer: str = "this function"):
    """Return imported pandas module or raise a helpful ImportError."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover - import time failure
        raise ImportError(
            f"Missing required dependency 'pandas' for {consumer}. Install with: pip install pandas"
        ) from e
    return pd


def get_bars_df(
    symbol: str,
    timeframe: str | Any = "1Min",
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    adjustment: str | None = None,
    feed: str | None = None,
) -> "pd.DataFrame":
    """Fetch bars for ``symbol`` and return a normalized DataFrame."""
    from alpaca.common.exceptions import APIError
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    _pd = _require_pandas("get_bars_df")
    rest = _get_rest()
    feed = feed or os.getenv('ALPACA_DATA_FEED', 'iex')
    adjustment = adjustment or os.getenv('ALPACA_ADJUSTMENT', 'all')
    tf_raw = timeframe
    tf = _normalize_timeframe_for_tradeapi(tf_raw)
    if start is None or end is None:
        try:
            base_tf = tf_raw if isinstance(tf_raw, TimeFrame) else TimeFrame(1, TimeFrameUnit.Day)
        except (ValueError, TypeError):
            base_tf = TimeFrame(1, TimeFrameUnit.Day)
        start, end = _bars_time_window(base_tf)
    start_s, end_s = _format_start_end_for_tradeapi(tf, start, end)
    try:
        df = rest.get_bars(symbol, timeframe=tf, start=start_s, end=end_s, adjustment=adjustment, feed=feed, limit=None).df
        if isinstance(df, _pd.DataFrame) and (not df.empty):
            return df.reset_index(drop=False)
        return _pd.DataFrame()
    except APIError as e:
        req = {'timeframe_raw': str(tf_raw), 'timeframe_norm': tf, 'feed': feed, 'start': start_s, 'end': end_s, 'adjustment': adjustment}
        body = ''
        try:
            body = e.response.text
        except (ValueError, TypeError):
            pass
        _log.error('ALPACA_FAIL', extra={'symbol': symbol, 'timeframe': tf, 'feed': feed, 'start': start_s, 'end': end_s, 'status_code': getattr(e, 'status_code', None), 'endpoint': 'alpaca/bars', 'query_params': req, 'body': body})
        return _pd.DataFrame()


class AlpacaOrderError(RuntimeError):
    """Base exception for order submission issues."""


class AlpacaOrderHTTPError(AlpacaOrderError):
    def __init__(self, status_code: int, message: str, payload: Optional[dict[str, Any]] | None = None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.payload = payload or {}


class AlpacaOrderNetworkError(AlpacaOrderError):
    pass


@dataclass(frozen=True)
class _AlpacaConfig:
    base_url: str
    key_id: str | None
    secret_key: str | None
    shadow: bool

    @staticmethod
    def from_env() -> "_AlpacaConfig":
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        sec = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        shadow_env = os.getenv("ALPACA_SHADOW", "")
        shadow = is_shadow_mode() or str(shadow_env).strip().lower() in {"1", "true", "yes", "on"}
        return _AlpacaConfig(base, key, sec, shadow)


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _as_int(n: Any) -> int:
    return int(round(float(n)))


def _to_public_dict(resp_json: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "id",
        "client_order_id",
        "symbol",
        "qty",
        "side",
        "type",
        "time_in_force",
        "limit_price",
        "stop_price",
        "status",
        "created_at",
        "submitted_at",
        "filled_qty",
    ]
    return {k: resp_json.get(k) for k in fields if k in resp_json}


def _sdk_submit(
    client: Any,
    *,
    symbol: str,
    qty: int,
    side: str,
    type: str,
    time_in_force: str,
    limit_price: float | None,
    stop_price: float | None,
    idempotency_key: str | None,
    timeout: float | int | None,
) -> dict[str, Any]:
    submit = getattr(client, "submit_order", None)
    if submit is None:
        raise AttributeError("client.submit_order is not available")

    kwargs: dict[str, Any] = dict(
        symbol=symbol,
        qty=str(qty),
        side=side,
        type=type,
        time_in_force=time_in_force,
    )
    if limit_price is not None:
        kwargs["limit_price"] = str(limit_price)
    if stop_price is not None:
        kwargs["stop_price"] = str(stop_price)
    if idempotency_key:
        kwargs["client_order_id"] = idempotency_key

    order = submit(**kwargs)
    if hasattr(order, "_raw"):
        data = dict(order._raw)  # type: ignore[attr-defined]
    elif hasattr(order, "__dict__"):
        data = dict(order.__dict__)
    else:
        try:
            data = json.loads(json.dumps(order))
        except Exception:
            data = {
                "id": getattr(order, "id", None),
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "status": getattr(order, "status", None),
            }
    return _to_public_dict(data)


def _http_submit(
    cfg: _AlpacaConfig,
    *,
    symbol: str,
    qty: int,
    side: str,
    type: str,
    time_in_force: str,
    limit_price: float | None,
    stop_price: float | None,
    idempotency_key: str | None,
    timeout: float | int | None,
) -> dict[str, Any]:
    url = f"{cfg.base_url}/v2/orders"
    headers = {
        "APCA-API-KEY-ID": cfg.key_id or "",
        "APCA-API-SECRET-KEY": cfg.secret_key or "",
        "Content-Type": "application/json",
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    payload: dict[str, Any] = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
    }
    if limit_price is not None:
        payload["limit_price"] = str(limit_price)
    if stop_price is not None:
        payload["stop_price"] = str(stop_price)

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout or 10)
    except requests.RequestException as e:  # pragma: no cover - network error path
        raise AlpacaOrderNetworkError(f"Network error calling {url}: {e}") from e

    try:
        content: dict[str, Any] | None = resp.json()
    except Exception:
        content = None

    if resp.status_code >= 400:
        msg = "rate limited" if resp.status_code == 429 else (content or {}).get("message") or resp.text
        raise AlpacaOrderHTTPError(resp.status_code, msg, payload=content or {})

    return _to_public_dict(content or {})


def submit_order(
    symbol: str,
    qty: int | float | str,
    side: str,
    type: str = "market",
    time_in_force: str = "day",
    *,
    limit_price: float | None = None,
    stop_price: float | None = None,
    shadow: bool | None = None,
    timeout: float | int | None = None,
    idempotency_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Submit an order via Alpaca SDK or HTTP.

    - Supports shadow mode for offline testing.
    - Raises :class:`AttributeError` if a provided client lacks ``submit_order``.
    - Maps HTTP errors (incl. 429) to :class:`AlpacaOrderHTTPError`.
    """
    # AI-AGENT-REF: expose deterministic submit_order with shadow + HTTP fallback
    cfg = _AlpacaConfig.from_env()
    do_shadow = cfg.shadow if shadow is None else bool(shadow)
    q_int = _as_int(qty)

    if do_shadow:
        oid = f"shadow-{uuid.uuid4().hex[:16]}"
        return {
            "id": oid,
            "client_order_id": idempotency_key or oid,
            "symbol": symbol,
            "qty": str(q_int),
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
            "status": "accepted",
            "submitted_at": _ts(),
            "filled_qty": "0",
        }

    if client is not None:
        return _sdk_submit(
            client,
            symbol=symbol,
            qty=q_int,
            side=side,
            type=type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            idempotency_key=idempotency_key,
            timeout=timeout,
        )

    try:
        from alpaca.trading.client import TradingClient as _TradingClient
        rest = _TradingClient(
            cfg.key_id,
            cfg.secret_key,
            base_url=cfg.base_url,
        )
        return _sdk_submit(
            rest,
            symbol=symbol,
            qty=q_int,
            side=side,
            type=type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            idempotency_key=idempotency_key,
            timeout=timeout,
        )
    except ModuleNotFoundError:
        pass

    return _http_submit(
        cfg,
        symbol=symbol,
        qty=q_int,
        side=side,
        type=type,
        time_in_force=time_in_force,
        limit_price=limit_price,
        stop_price=stop_price,
        idempotency_key=idempotency_key,
        timeout=timeout,
    )

def alpaca_get(*_a, **_k):
    return None

def start_trade_updates_stream(*_a, **_k):
    return None
__all__ = [
    'is_shadow_mode',
    'RETRY_HTTP_CODES',
    'RETRYABLE_HTTP_STATUSES',
    'submit_order',
    'AlpacaOrderError',
    'AlpacaOrderHTTPError',
    'AlpacaOrderNetworkError',
    'generate_client_order_id',
    '_bars_time_window',
    'get_bars_df',
    'alpaca_get',
    'start_trade_updates_stream',
]
