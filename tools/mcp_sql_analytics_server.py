"""Read-only SQL analytics MCP server for runtime execution/trade history."""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))

_DEFAULT_RUNTIME_ROOT = Path("/var/lib/ai-trading-bot/runtime")


def _runtime_root(args: dict[str, Any]) -> Path:
    raw = str(args.get("runtime_root") or _DEFAULT_RUNTIME_ROOT)
    return Path(raw).expanduser().resolve()


def _trade_history_path(args: dict[str, Any]) -> Path:
    root = _runtime_root(args)
    raw = str(args.get("trade_history_path") or "trade_history.parquet").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _load_trade_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"trade history not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        raise RuntimeError(f"unsupported trade history format: {path.suffix}")
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("trade history payload was not a DataFrame")
    return frame


def _float_series(frame: pd.DataFrame, name: str, aliases: list[str]) -> pd.Series:
    for candidate in [name, *aliases]:
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce")
    return pd.Series([None] * len(frame), index=frame.index, dtype="float64")


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    for candidate in ("filled_at", "fill_ts", "timestamp", "submitted_at", "created_at"):
        if candidate in frame.columns:
            return pd.to_datetime(frame[candidate], errors="coerce", utc=True)
    return pd.Series([pd.NaT] * len(frame), index=frame.index, dtype="datetime64[ns, UTC]")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        return prepared

    prepared["slippage_bps_norm"] = _float_series(
        prepared,
        "slippage_bps",
        aliases=["realized_slippage_bps", "mean_slippage_bps"],
    )
    prepared["capture_ratio_norm"] = _float_series(
        prepared,
        "execution_capture_ratio",
        aliases=["capture_ratio"],
    )
    ts = _timestamp_series(prepared)
    prepared["event_ts_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    prepared["event_hour_utc"] = ts.dt.hour

    for column in prepared.columns:
        if str(prepared[column].dtype).startswith("datetime64"):
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce", utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
    return prepared


def _validate_read_only_query(query: str) -> str:
    raw = query.strip()
    if not raw:
        raise RuntimeError("query is required")
    trimmed = raw.rstrip(";").strip()
    if ";" in trimmed:
        raise RuntimeError("multiple statements are not allowed")
    lowered = trimmed.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise RuntimeError("only SELECT/WITH read-only queries are allowed")
    blocked = [
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " create ",
        " attach ",
        " pragma ",
        " replace ",
        " vacuum ",
    ]
    haystack = f" {lowered} "
    for token in blocked:
        if token in haystack:
            raise RuntimeError("query contains blocked keyword for read-only mode")
    return trimmed


def _query_sql(frame: pd.DataFrame, query: str, limit: int) -> dict[str, Any]:
    safe_query = _validate_read_only_query(query)
    max_rows = max(1, int(limit))
    with sqlite3.connect(":memory:") as conn:
        frame.to_sql("trade_history", conn, index=False, if_exists="replace")
        conn.execute("PRAGMA query_only = ON")
        wrapped = f"SELECT * FROM ({safe_query}) LIMIT {max_rows}"
        cursor = conn.execute(wrapped)
        columns = [str(desc[0]) for desc in cursor.description or []]
        rows = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
    return {"columns": columns, "rows": rows, "row_count": len(rows)}


def tool_warehouse_status(args: dict[str, Any]) -> dict[str, Any]:
    path = _trade_history_path(args)
    frame = _prepare_frame(_load_trade_history(path))
    return {
        "trade_history_path": str(path),
        "table": "trade_history",
        "rows": int(len(frame)),
        "columns": [str(col) for col in frame.columns],
        "derived_columns": ["slippage_bps_norm", "capture_ratio_norm", "event_ts_utc", "event_hour_utc"],
    }


def tool_query_trade_history_sql(args: dict[str, Any]) -> dict[str, Any]:
    path = _trade_history_path(args)
    query = str(args.get("query") or "").strip()
    limit = int(args.get("limit") or 250)
    frame = _prepare_frame(_load_trade_history(path))
    result = _query_sql(frame, query=query, limit=limit)
    return {
        "trade_history_path": str(path),
        "table": "trade_history",
        "query": query,
        "limit": max(1, limit),
        **result,
    }


def tool_execution_trend_examples(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "table": "trade_history",
        "examples": [
            {
                "name": "slippage_by_symbol",
                "query": (
                    "SELECT symbol, AVG(slippage_bps_norm) AS mean_slippage_bps, "
                    "COUNT(*) AS fills "
                    "FROM trade_history "
                    "WHERE slippage_bps_norm IS NOT NULL "
                    "GROUP BY symbol "
                    "ORDER BY mean_slippage_bps DESC"
                ),
            },
            {
                "name": "capture_by_hour_utc",
                "query": (
                    "SELECT event_hour_utc, AVG(capture_ratio_norm) AS mean_capture_ratio, "
                    "COUNT(*) AS fills "
                    "FROM trade_history "
                    "WHERE capture_ratio_norm IS NOT NULL AND event_hour_utc IS NOT NULL "
                    "GROUP BY event_hour_utc "
                    "ORDER BY event_hour_utc"
                ),
            },
            {
                "name": "reject_rate_by_symbol",
                "query": (
                    "SELECT symbol, AVG(CASE WHEN status IN ('rejected', 'REJECTED') THEN 1.0 ELSE 0.0 END) "
                    "AS reject_rate, COUNT(*) AS orders "
                    "FROM trade_history "
                    "GROUP BY symbol ORDER BY reject_rate DESC"
                ),
            },
        ],
        "notes": [
            "Only SELECT/WITH statements are allowed.",
            "Use derived columns slippage_bps_norm, capture_ratio_norm, event_hour_utc.",
        ],
    }


TOOLS = {
    "warehouse_status": tool_warehouse_status,
    "query_trade_history_sql": tool_query_trade_history_sql,
    "execution_trend_examples": tool_execution_trend_examples,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "warehouse_status",
        "description": "Describe read-only SQL warehouse table built from runtime trade history.",
    },
    {
        "name": "query_trade_history_sql",
        "description": "Execute read-only SQL over trade_history table (SELECT/WITH only).",
    },
    {
        "name": "execution_trend_examples",
        "description": "Return ready-to-run SQL examples for slippage/capture/reject analytics.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="SQL analytics MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_sql_analytics",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
