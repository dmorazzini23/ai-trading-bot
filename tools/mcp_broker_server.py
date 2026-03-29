"""Broker MCP-style server (read-only controls)."""

from __future__ import annotations

import importlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))

_DEFAULT_RUNTIME_ROOT = Path("/var/lib/ai-trading-bot/runtime")


def _alpaca_base_url() -> str:
    return os.getenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def _alpaca_headers() -> dict[str, str]:
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError("missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def _alpaca_get(path: str, query: dict[str, Any] | None = None) -> dict[str, Any]:
    base = _alpaca_base_url()
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    request = urllib.request.Request(url=url, method="GET")
    for key, value in _alpaca_headers().items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)
            return {"url": url, "status_code": int(response.status), "payload": payload}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"url": url, "status_code": int(exc.code), "error": body}
    except urllib.error.URLError as exc:
        return {"url": url, "status_code": None, "error": str(exc.reason)}


def _runtime_root(args: dict[str, Any]) -> Path:
    raw = str(args.get("runtime_root") or _DEFAULT_RUNTIME_ROOT)
    return Path(raw).expanduser().resolve()


def tool_alpaca_account(_: dict[str, Any]) -> dict[str, Any]:
    return _alpaca_get("/v2/account")


def tool_alpaca_positions(args: dict[str, Any]) -> dict[str, Any]:
    payload = _alpaca_get("/v2/positions")
    data = payload.get("payload")
    if isinstance(data, list):
        payload["position_count"] = len(data)
    return payload


def tool_alpaca_open_orders(args: dict[str, Any]) -> dict[str, Any]:
    status = str(args.get("status") or "open")
    limit = int(args.get("limit") or 200)
    payload = _alpaca_get("/v2/orders", {"status": status, "limit": limit})
    data = payload.get("payload")
    if isinstance(data, list):
        payload["order_count"] = len(data)
    return payload


def tool_alpaca_clock(_: dict[str, Any]) -> dict[str, Any]:
    return _alpaca_get("/v2/clock")


def tool_runtime_broker_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    root = _runtime_root(args)
    report = root / "runtime_performance_report_latest.json"
    if not report.exists():
        return {"exists": False, "path": str(report)}
    payload = json.loads(report.read_text(encoding="utf-8"))
    trade = payload.get("trade_history") or {}
    snapshot = {
        "exists": True,
        "path": str(report),
        "broker_open_position_count": trade.get("broker_open_position_count"),
        "broker_open_positions_available": trade.get("broker_open_positions_available"),
        "open_position_reconciliation": trade.get("open_position_reconciliation"),
    }
    return snapshot


TOOLS = {
    "alpaca_account": tool_alpaca_account,
    "alpaca_positions": tool_alpaca_positions,
    "alpaca_open_orders": tool_alpaca_open_orders,
    "alpaca_clock": tool_alpaca_clock,
    "runtime_broker_snapshot": tool_runtime_broker_snapshot,
}

TOOL_SPECS: list[ToolSpec] = [
    {"name": "alpaca_account", "description": "Read Alpaca account state (GET /v2/account)."},
    {"name": "alpaca_positions", "description": "Read Alpaca positions (GET /v2/positions)."},
    {"name": "alpaca_open_orders", "description": "Read Alpaca open orders (GET /v2/orders)."},
    {"name": "alpaca_clock", "description": "Read Alpaca market clock (GET /v2/clock)."},
    {
        "name": "runtime_broker_snapshot",
        "description": "Extract broker/reconciliation snapshot from runtime report JSON.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Broker MCP-style server (read-only)",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_broker",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
