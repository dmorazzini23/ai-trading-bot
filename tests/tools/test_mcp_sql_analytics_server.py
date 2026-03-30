from __future__ import annotations

import pandas as pd

from tools import mcp_sql_analytics_server as sql_srv


def test_query_trade_history_sql_blocks_mutation(monkeypatch) -> None:
    frame = pd.DataFrame({"symbol": ["AAPL"], "slippage_bps": [1.2]})
    monkeypatch.setattr(sql_srv, "_load_trade_history", lambda path: frame)

    try:
        sql_srv.tool_query_trade_history_sql({"query": "UPDATE trade_history SET symbol='MSFT'"})
    except RuntimeError as exc:
        assert "read-only" in str(exc).lower()
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError for mutating query")


def test_query_trade_history_sql_returns_rows(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "slippage_bps": [1.0, 2.0, 3.0],
            "execution_capture_ratio": [0.2, 0.3, 0.4],
            "filled_at": [
                "2026-03-29T14:01:00Z",
                "2026-03-29T14:02:00Z",
                "2026-03-29T15:01:00Z",
            ],
        }
    )
    monkeypatch.setattr(sql_srv, "_load_trade_history", lambda path: frame)

    payload = sql_srv.tool_query_trade_history_sql(
        {
            "query": (
                "SELECT symbol, AVG(slippage_bps_norm) AS mean_slippage "
                "FROM trade_history GROUP BY symbol ORDER BY symbol"
            )
        }
    )
    assert payload["row_count"] == 2
    assert payload["rows"][0]["symbol"] == "AAPL"
    assert float(payload["rows"][0]["mean_slippage"]) == 1.5
