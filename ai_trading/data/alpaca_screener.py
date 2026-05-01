from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from threading import Lock
from typing import Any, cast
from zoneinfo import ZoneInfo

from ai_trading.broker.alpaca_credentials import alpaca_auth_headers
from ai_trading.config.management import get_env
from ai_trading.logging import logger
from ai_trading.utils.market_calendar import is_trading_day, previous_trading_session
from ai_trading.utils.env import get_alpaca_data_base_url
from ai_trading.utils.http import clamp_request_timeout, get as http_get

_NY = ZoneInfo("America/New_York")
_MARKET_OPEN = time(hour=9, minute=30)
_CACHE_LOCK = Lock()
_CACHE: dict[
    tuple[str, int, str],
    tuple[float, "MarketMoversSnapshot | MostActivesSnapshot"],
] = {}
_LAST_GOOD: dict[tuple[str, int, str], "MarketMoversSnapshot | MostActivesSnapshot"] = {}
_LAST_GOOD_AT: dict[tuple[str, int, str], float] = {}


@dataclass(frozen=True)
class MarketMover:
    symbol: str
    percent_change: float
    change: float
    price: float


@dataclass(frozen=True)
class MarketMoversSnapshot:
    gainers: list[MarketMover]
    losers: list[MarketMover]
    market_type: str
    last_updated: datetime
    used_fallback: bool = False


@dataclass(frozen=True)
class ActiveStock:
    symbol: str
    volume: float
    trade_count: float


@dataclass(frozen=True)
class MostActivesSnapshot:
    most_actives: list[ActiveStock]
    last_updated: datetime
    used_fallback: bool = False


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _cache_ttl_seconds(override: int | None = None) -> int:
    if override is not None:
        return max(0, int(override))
    return max(
        0,
        int(get_env("AI_TRADING_DYNAMIC_UNIVERSE_REFRESH_SEC", 300, cast=int)),
    )


def _parse_datetime(raw_value: Any, *, default: datetime | None = None) -> datetime:
    if isinstance(raw_value, datetime):
        if raw_value.tzinfo is None:
            return raw_value.replace(tzinfo=UTC)
        return raw_value.astimezone(UTC)
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if value:
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=UTC)
                return parsed.astimezone(UTC)
    return default or _now_utc()


def _safe_float(raw_value: Any, default: float = 0.0) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _current_market_day(now: datetime | None = None) -> date:
    current = (now or _now_utc()).astimezone(_NY)
    market_day = current.date()
    if current.time() < _MARKET_OPEN or not is_trading_day(market_day):
        return previous_trading_session(market_day)
    return market_day


def _snapshot_market_day(last_updated: datetime) -> date:
    return last_updated.astimezone(_NY).date()


def _cache_key(kind: str, top: int, market_type: str) -> tuple[str, int, str]:
    return (kind, max(1, int(top)), str(market_type or "stocks").strip().lower() or "stocks")


def _cache_valid(
    snapshot: MarketMoversSnapshot | MostActivesSnapshot,
    fetched_at: float,
    ttl_seconds: int,
    *,
    now: datetime | None = None,
) -> bool:
    if ttl_seconds <= 0:
        return False
    age = max(0.0, _now_utc().timestamp() - fetched_at)
    if age > float(ttl_seconds):
        return False
    last_updated = getattr(snapshot, "last_updated", None)
    if isinstance(last_updated, datetime):
        return _snapshot_market_day(last_updated) == _current_market_day(now)
    return True


def _last_good_max_age_seconds() -> int:
    return max(
        0,
        int(
            get_env(
                "AI_TRADING_SCREENER_LAST_GOOD_MAX_AGE_SEC",
                _cache_ttl_seconds(),
                cast=int,
            )
        ),
    )


def _last_good_valid(
    snapshot: MarketMoversSnapshot | MostActivesSnapshot,
    fetched_at: float | None,
    *,
    now: datetime | None = None,
) -> bool:
    if fetched_at is None:
        return False
    max_age = _last_good_max_age_seconds()
    if max_age <= 0:
        return False
    return _cache_valid(snapshot, fetched_at, max_age, now=now)


def _movers_url(market_type: str) -> str:
    normalized = str(market_type or "stocks").strip().lower() or "stocks"
    return f"{get_alpaca_data_base_url().rstrip('/')}/v1beta1/screener/{normalized}/movers"


def _most_actives_url(market_type: str) -> str:
    normalized = str(market_type or "stocks").strip().lower() or "stocks"
    return f"{get_alpaca_data_base_url().rstrip('/')}/v1beta1/screener/{normalized}/most-actives"


def _request_json(url: str, *, top: int, by: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"top": max(1, int(top))}
    if by:
        params["by"] = str(by).strip().lower()
    response = http_get(
        url,
        params=params,
        headers=alpaca_auth_headers(),
        timeout=clamp_request_timeout((3.0, 10.0)),
    )
    status_code = int(getattr(response, "status_code", 599) or 599)
    if status_code >= 400:
        raise RuntimeError(f"screener_http_{status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("screener_payload_invalid")
    return payload


def _parse_mover_row(raw_row: Any) -> MarketMover | None:
    if not isinstance(raw_row, dict):
        return None
    symbol = str(raw_row.get("symbol", "")).strip().upper()
    if not symbol:
        return None
    return MarketMover(
        symbol=symbol,
        percent_change=_safe_float(raw_row.get("percent_change")),
        change=_safe_float(raw_row.get("change")),
        price=_safe_float(raw_row.get("price")),
    )


def _parse_active_row(raw_row: Any) -> ActiveStock | None:
    if not isinstance(raw_row, dict):
        return None
    symbol = str(raw_row.get("symbol", "")).strip().upper()
    if not symbol:
        return None
    return ActiveStock(
        symbol=symbol,
        volume=_safe_float(raw_row.get("volume")),
        trade_count=_safe_float(raw_row.get("trade_count")),
    )


def _clone_market_movers(snapshot: MarketMoversSnapshot, *, used_fallback: bool) -> MarketMoversSnapshot:
    return MarketMoversSnapshot(
        gainers=list(snapshot.gainers),
        losers=list(snapshot.losers),
        market_type=snapshot.market_type,
        last_updated=snapshot.last_updated,
        used_fallback=used_fallback,
    )


def _clone_most_actives(snapshot: MostActivesSnapshot, *, used_fallback: bool) -> MostActivesSnapshot:
    return MostActivesSnapshot(
        most_actives=list(snapshot.most_actives),
        last_updated=snapshot.last_updated,
        used_fallback=used_fallback,
    )


def fetch_market_movers(
    top: int = 10,
    *,
    market_type: str = "stocks",
    ttl_seconds: int | None = None,
    now: datetime | None = None,
) -> MarketMoversSnapshot:
    key = _cache_key("movers", top, market_type)
    ttl = _cache_ttl_seconds(ttl_seconds)
    with _CACHE_LOCK:
        cached_entry = _CACHE.get(key)
        if cached_entry is not None:
            fetched_at, snapshot = cached_entry
            if _cache_valid(snapshot, fetched_at, ttl, now=now):
                return cast(MarketMoversSnapshot, snapshot)
    url = _movers_url(market_type)
    try:
        payload = _request_json(url, top=top)
        snapshot = MarketMoversSnapshot(
            gainers=[
                mover
                for raw_row in payload.get("gainers", [])
                if (mover := _parse_mover_row(raw_row)) is not None
            ],
            losers=[
                mover
                for raw_row in payload.get("losers", [])
                if (mover := _parse_mover_row(raw_row)) is not None
            ],
            market_type=str(payload.get("market_type", market_type or "stocks")).strip().lower() or "stocks",
            last_updated=_parse_datetime(payload.get("last_updated"), default=now or _now_utc()),
            used_fallback=False,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.warning(
            "ALPACA_SCREENER_REFRESH_FAILED",
            extra={
                "kind": "market_movers",
                "top": max(1, int(top)),
                "market_type": str(market_type or "stocks"),
                "detail": str(exc),
            },
        )
        with _CACHE_LOCK:
            fallback = _LAST_GOOD.get(key)
            fallback_at = _LAST_GOOD_AT.get(key)
        if isinstance(fallback, MarketMoversSnapshot) and _last_good_valid(fallback, fallback_at, now=now):
            logger.info(
                "ALPACA_SCREENER_STALE_CACHE_USED",
                extra={"kind": "market_movers", "top": max(1, int(top))},
            )
            return _clone_market_movers(fallback, used_fallback=True)
        return MarketMoversSnapshot(
            gainers=[],
            losers=[],
            market_type=str(market_type or "stocks").strip().lower() or "stocks",
            last_updated=now or _now_utc(),
            used_fallback=True,
        )
    fetched_at = _now_utc().timestamp()
    with _CACHE_LOCK:
        _CACHE[key] = (fetched_at, snapshot)
        _LAST_GOOD[key] = snapshot
        _LAST_GOOD_AT[key] = fetched_at
    logger.info(
        "ALPACA_SCREENER_REFRESHED",
        extra={
            "kind": "market_movers",
            "top": max(1, int(top)),
            "gainers": len(snapshot.gainers),
            "losers": len(snapshot.losers),
            "market_type": snapshot.market_type,
            "last_updated": snapshot.last_updated.isoformat(),
        },
    )
    return cast(MarketMoversSnapshot, snapshot)


def fetch_most_actives(
    top: int = 10,
    *,
    by: str = "volume",
    market_type: str = "stocks",
    ttl_seconds: int | None = None,
    now: datetime | None = None,
) -> MostActivesSnapshot:
    key = _cache_key("most_actives", top, f"{market_type}:{by}")
    ttl = _cache_ttl_seconds(ttl_seconds)
    with _CACHE_LOCK:
        cached_entry = _CACHE.get(key)
        if cached_entry is not None:
            fetched_at, snapshot = cached_entry
            if _cache_valid(snapshot, fetched_at, ttl, now=now):
                return cast(MostActivesSnapshot, snapshot)
    url = _most_actives_url(market_type)
    try:
        payload = _request_json(url, top=top, by=by)
        snapshot = MostActivesSnapshot(
            most_actives=[
                active
                for raw_row in payload.get("most_actives", [])
                if (active := _parse_active_row(raw_row)) is not None
            ],
            last_updated=_parse_datetime(payload.get("last_updated"), default=now or _now_utc()),
            used_fallback=False,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.warning(
            "ALPACA_SCREENER_REFRESH_FAILED",
            extra={
                "kind": "most_actives",
                "top": max(1, int(top)),
                "market_type": str(market_type or "stocks"),
                "detail": str(exc),
            },
        )
        with _CACHE_LOCK:
            fallback = _LAST_GOOD.get(key)
            fallback_at = _LAST_GOOD_AT.get(key)
        if isinstance(fallback, MostActivesSnapshot) and _last_good_valid(fallback, fallback_at, now=now):
            logger.info(
                "ALPACA_SCREENER_STALE_CACHE_USED",
                extra={"kind": "most_actives", "top": max(1, int(top))},
            )
            return _clone_most_actives(fallback, used_fallback=True)
        return MostActivesSnapshot(
            most_actives=[],
            last_updated=now or _now_utc(),
            used_fallback=True,
        )
    fetched_at = _now_utc().timestamp()
    with _CACHE_LOCK:
        _CACHE[key] = (fetched_at, snapshot)
        _LAST_GOOD[key] = snapshot
        _LAST_GOOD_AT[key] = fetched_at
    logger.info(
        "ALPACA_SCREENER_REFRESHED",
        extra={
            "kind": "most_actives",
            "top": max(1, int(top)),
            "count": len(snapshot.most_actives),
            "last_updated": snapshot.last_updated.isoformat(),
        },
    )
    return cast(MostActivesSnapshot, snapshot)


def reset_screener_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        _LAST_GOOD.clear()
        _LAST_GOOD_AT.clear()


def snapshot_to_dict(snapshot: MarketMoversSnapshot | MostActivesSnapshot) -> dict[str, Any]:
    payload = cast(dict[str, Any], asdict(snapshot))
    last_updated = payload.get("last_updated")
    if isinstance(last_updated, datetime):
        payload["last_updated"] = last_updated.isoformat()
    return payload


__all__ = [
    "ActiveStock",
    "MarketMover",
    "MarketMoversSnapshot",
    "MostActivesSnapshot",
    "fetch_market_movers",
    "fetch_most_actives",
    "reset_screener_cache",
    "snapshot_to_dict",
]
