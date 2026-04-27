from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools import mcp_sql_analytics_server as sql_srv


def test_query_trade_history_sql_blocks_mutation(monkeypatch) -> None:
    frame = pd.DataFrame({"symbol": ["AAPL"], "slippage_bps": [1.2]})
    monkeypatch.setattr(sql_srv, "_load_trade_history", lambda path, **_: frame)

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
    monkeypatch.setattr(sql_srv, "_load_trade_history", lambda path, **_: frame)

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


def test_trade_history_path_must_stay_within_runtime_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.parquet"

    with pytest.raises(RuntimeError, match="within runtime root"):
        sql_srv._trade_history_path(
            {
                "runtime_root": str(tmp_path),
                "trade_history_path": str(outside),
            }
        )


def test_pickle_trade_history_requires_explicit_trust(tmp_path: Path) -> None:
    path = tmp_path / "trade_history.pkl"
    pd.DataFrame({"symbol": ["AAPL"]}).to_pickle(path)

    with pytest.raises(RuntimeError, match="allow_trusted_pickle"):
        sql_srv.tool_warehouse_status(
            {
                "runtime_root": str(tmp_path),
                "trade_history_path": "trade_history.pkl",
            }
        )


def test_pickle_trade_history_allowed_when_explicitly_trusted(tmp_path: Path) -> None:
    path = tmp_path / "trade_history.pkl"
    pd.DataFrame({"symbol": ["AAPL"]}).to_pickle(path)

    payload = sql_srv.tool_warehouse_status(
        {
            "runtime_root": str(tmp_path),
            "trade_history_path": "trade_history.pkl",
            "allow_trusted_pickle": True,
        }
    )

    assert payload["rows"] == 1
    assert payload["trade_history_path"] == str(path)
