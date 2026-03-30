"""Market events MCP server for calendar-aware risk warnings."""

from __future__ import annotations

import importlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, cast

import pandas_market_calendars as mcal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))

_HIGH_IMPACT_KEYWORDS = (
    "fomc",
    "rate decision",
    "fed",
    "cpi",
    "ppi",
    "nonfarm",
    "nfp",
    "earnings",
    "gdp",
)


def _http_get_json(url: str, timeout_s: float = 8.0) -> Any:
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"market events endpoint HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"market events endpoint request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("market events endpoint returned invalid JSON") from exc


def _now_ts(args: dict[str, Any]) -> int:
    raw = args.get("now_ts")
    if raw is None:
        return int(time.time())
    return int(float(raw))


def _parse_iso_ts(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _coerce_symbols(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    if isinstance(value, str):
        chunks = [part.strip().upper() for part in value.split(",") if part.strip()]
        return chunks
    return []


def _impact_level(event: dict[str, Any]) -> str:
    raw = str(event.get("impact") or event.get("severity") or "").strip().lower()
    if raw in {"high", "critical"}:
        return "high"
    if raw in {"medium", "moderate", "warning"}:
        return "medium"
    if raw in {"low", "info"}:
        return "low"
    title = str(event.get("title") or "").strip().lower()
    if any(token in title for token in _HIGH_IMPACT_KEYWORDS):
        return "high"
    return "unknown"


def _normalize_event(item: dict[str, Any]) -> dict[str, Any]:
    title = (
        str(item.get("title") or "").strip()
        or str(item.get("name") or "").strip()
        or str(item.get("event") or "").strip()
        or "untitled_event"
    )
    ts = (
        _parse_iso_ts(item.get("timestamp"))
        or _parse_iso_ts(item.get("datetime"))
        or _parse_iso_ts(item.get("time"))
        or _parse_iso_ts(item.get("date"))
    )
    source = str(item.get("source") or item.get("provider") or "external").strip()
    event_type = str(item.get("type") or item.get("category") or "event").strip()
    symbols = _coerce_symbols(item.get("symbols") or item.get("tickers"))

    normalized = {
        "title": title,
        "timestamp": ts,
        "timestamp_iso": (
            datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")
            if ts is not None
            else None
        ),
        "impact": str(item.get("impact") or item.get("severity") or "").strip() or None,
        "impact_level": _impact_level(item),
        "type": event_type,
        "source": source,
        "symbols": symbols,
        "raw": item,
    }
    return normalized


def _events_url(args: dict[str, Any]) -> str:
    direct = str(args.get("url") or "").strip()
    if direct:
        return direct
    return str(os.getenv("AI_TRADING_MARKET_EVENTS_JSON_URL") or "").strip()


def tool_market_sessions(args: dict[str, Any]) -> dict[str, Any]:
    calendar_name = str(args.get("calendar") or "XNYS").strip().upper()
    days = max(1, int(args.get("days") or 5))
    raw_start = str(args.get("start_date") or "").strip()
    if raw_start:
        start_day = date.fromisoformat(raw_start)
    else:
        start_day = datetime.now(UTC).date()
    end_day = start_day + timedelta(days=days + 3)

    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start_day.isoformat(), end_date=end_day.isoformat())
    sessions: list[dict[str, Any]] = []
    for session_date, row in schedule.head(days).iterrows():
        market_open = row["market_open"].to_pydatetime().astimezone(UTC)
        market_close = row["market_close"].to_pydatetime().astimezone(UTC)
        duration_hours = (market_close - market_open).total_seconds() / 3600.0
        sessions.append(
            {
                "session_date": str(session_date.date()),
                "market_open_utc": market_open.isoformat().replace("+00:00", "Z"),
                "market_close_utc": market_close.isoformat().replace("+00:00", "Z"),
                "duration_hours": duration_hours,
                "is_early_close": duration_hours < 6.0,
            }
        )
    return {
        "calendar": calendar_name,
        "start_date": start_day.isoformat(),
        "requested_days": days,
        "count": len(sessions),
        "sessions": sessions,
    }


def tool_fetch_events(args: dict[str, Any]) -> dict[str, Any]:
    url = _events_url(args)
    if not url:
        return {
            "configured": False,
            "reason": "missing_market_events_url",
            "env_key": "AI_TRADING_MARKET_EVENTS_JSON_URL",
            "events": [],
            "count": 0,
        }

    timeout_s = float(args.get("timeout_s") or 8.0)
    raw = _http_get_json(url=url, timeout_s=timeout_s)
    if isinstance(raw, dict):
        rows = raw.get("events")
        if not isinstance(rows, list):
            rows = raw.get("data")
        if not isinstance(rows, list):
            rows = []
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(_normalize_event(row))
    normalized.sort(key=lambda item: int(item.get("timestamp") or 0))

    limit = max(1, int(args.get("limit") or 250))
    return {
        "configured": True,
        "url": url,
        "count": len(normalized[:limit]),
        "events": normalized[:limit],
    }


def tool_market_risk_window(args: dict[str, Any]) -> dict[str, Any]:
    horizon_hours = max(1, int(args.get("horizon_hours") or 24))
    now_ts = _now_ts(args)
    end_ts = now_ts + horizon_hours * 3600

    events_payload = tool_fetch_events(args)
    events = cast(list[dict[str, Any]], events_payload.get("events") or [])
    upcoming: list[dict[str, Any]] = []
    for event in events:
        ts = event.get("timestamp")
        if ts is None:
            continue
        event_ts = int(ts)
        if now_ts <= event_ts <= end_ts:
            upcoming.append(event)
    high_impact = [evt for evt in upcoming if str(evt.get("impact_level")) == "high"]
    medium_impact = [evt for evt in upcoming if str(evt.get("impact_level")) == "medium"]

    risk_level = "normal"
    if high_impact:
        risk_level = "high"
    elif medium_impact:
        risk_level = "elevated"

    recommendation = "normal_risk"
    if risk_level == "high":
        recommendation = "reduced_risk_and_tighter_exposure_limits"
    elif risk_level == "elevated":
        recommendation = "monitor_and_consider_smaller_position_sizes"

    sessions = tool_market_sessions(
        {
            "calendar": args.get("calendar") or "XNYS",
            "days": 2,
            "start_date": args.get("start_date"),
        }
    )
    return {
        "now_ts": now_ts,
        "horizon_hours": horizon_hours,
        "window_end_ts": end_ts,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "upcoming_count": len(upcoming),
        "high_impact_count": len(high_impact),
        "medium_impact_count": len(medium_impact),
        "upcoming_events": upcoming,
        "sessions": sessions.get("sessions"),
        "source_configured": bool(events_payload.get("configured", False)),
    }


def tool_market_events_source_status(args: dict[str, Any]) -> dict[str, Any]:
    url = _events_url(args)
    if not url:
        return {"configured": False, "url": None}
    parts = urllib.parse.urlparse(url)
    return {
        "configured": True,
        "url": url,
        "host": parts.hostname,
        "scheme": parts.scheme,
    }


TOOLS = {
    "market_sessions": tool_market_sessions,
    "fetch_events": tool_fetch_events,
    "market_risk_window": tool_market_risk_window,
    "market_events_source_status": tool_market_events_source_status,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "market_sessions",
        "description": "Return upcoming market sessions (open/close windows) for exchange calendar.",
    },
    {
        "name": "fetch_events",
        "description": "Fetch and normalize external market events feed JSON.",
    },
    {
        "name": "market_risk_window",
        "description": "Build upcoming-event risk window and suggested risk posture.",
    },
    {
        "name": "market_events_source_status",
        "description": "Show configured market-events feed endpoint metadata.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Market events MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_market_events",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
