from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path

import pytest

from ai_trading.strategies import backtester


pd = pytest.importorskip("pandas")


def _write_backtest_csv(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_time": [
                "2025-01-03T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-04T00:00:00Z",
                "2025-01-05T00:00:00Z",
                "2025-01-06T00:00:00Z",
                "2025-01-07T00:00:00Z",
                "2025-01-08T00:00:00Z",
                "2025-01-09T00:00:00Z",
                "2025-01-10T00:00:00Z",
                "2025-01-11T00:00:00Z",
            ],
            "open": [102, 100, 101, 101.5, 103, 104, 105, 106, 107, 108, 109, 110],
            "high": [102.5, 100.5, 101.5, 102.0, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5],
            "low": [101.5, 99.5, 100.5, 101.0, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "close": [102, 100, 101, 101.5, 103, 104, 105, 106, 107, 108, 109, 110],
        }
    )
    frame.to_csv(path, index=False)


def test_backtester_cli_writes_csv_and_json_artifacts(tmp_path: Path, caplog) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    json_path = tmp_path / "summary.json"
    data_dir.mkdir()
    _write_backtest_csv(data_dir / "AAPL.csv")

    with caplog.at_level(logging.INFO, logger="ai_trading.strategies.backtester"):
        backtester.main(
            [
                "--symbols",
                "AAPL",
                "--data-dir",
                str(data_dir),
                "--start",
                "2025-01-01",
                "--end",
                "2025-01-11",
                "--timestamp-col",
                "event_time",
                "--output-dir",
                str(out_dir),
                "--output-json",
                str(json_path),
            ]
        )

    assert (out_dir / "backtest_summary.csv").exists()
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "backtest_run_manifest.json").exists()
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    assert payload["artifact_type"] == "backtest_summary"
    assert payload["aggregate"]["symbols"] == 1
    assert payload["aggregate"]["total_bars"] == 11
    assert payload["config"]["timestamp_col"] == "event_time"
    assert payload["artifacts"]["summary_csv"] == str((out_dir / "backtest_summary.csv").resolve())
    assert payload["artifacts"]["trades_csv"] == str((out_dir / "trades.csv").resolve())
    assert payload["artifacts"]["manifest_json"] == str((out_dir / "backtest_run_manifest.json").resolve())
    assert payload["symbols"][0]["symbol"] == "AAPL"
    assert payload["symbols"][0]["bars"] == 11
    assert payload["inputs"]["symbols"]["AAPL"]["duplicate_timestamp_rows"] == 2
    assert payload["inputs"]["symbols"]["AAPL"]["missing_volume_filled"] is True

    manifest = json.loads((out_dir / "backtest_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["artifact_type"] == "backtest_run_manifest"
    assert manifest["inputs"]["source_files"]["AAPL"]["sha256"]
    assert manifest["outputs"]["summary_csv"]["sha256"]
    assert manifest["outputs"]["summary_json"]["sha256"] == hashlib.sha256(
        json_path.read_bytes()
    ).hexdigest()

    summary_df = pd.read_csv(out_dir / "backtest_summary.csv")
    assert set(summary_df.columns) >= {"symbol", "bars", "trades", "net_pnl", "sharpe"}
    assert any(record.msg == "BACKTEST_DUPLICATE_TIMESTAMPS_DEDUPED" for record in caplog.records)
    assert any(record.msg == "BACKTEST_ARTIFACTS_WRITTEN" for record in caplog.records)


def test_backtester_default_json_artifact_is_written(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_backtest_csv(data_dir / "MSFT.csv")

    backtester.main(
        [
            "--symbols",
            "MSFT",
            "--data-dir",
            str(data_dir),
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-11",
            "--timestamp-col",
            "event_time",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )

    summary_json = tmp_path / "runs" / "backtest_summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["config"]["symbols_loaded"] == ["MSFT"]
    assert (tmp_path / "runs" / "backtest_run_manifest.json").exists()


def test_backtest_engine_empty_input_returns_empty_result() -> None:
    engine = backtester.BacktestEngine(
        {"AAPL": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])},
        backtester.DefaultExecutionModel(),
    )

    result = engine.run(["AAPL"])

    assert result.trades.empty
    assert result.equity_curve.empty
    assert result.net_pnl == 0.0


def test_backtest_signal_uses_prior_bars_for_order_decision() -> None:
    engine = backtester.BacktestEngine(
        {"AAPL": pd.DataFrame()},
        backtester.DefaultExecutionModel(),
        initial_cash=1_000.0,
    )

    for close in [10.0, 10.0, 10.0, 10.0, 10.0]:
        assert engine._generate_orders_for_bar("AAPL", close) == []
    assert engine._generate_orders_for_bar("AAPL", 20.0) == []
    assert engine._generate_orders_for_bar("AAPL", 20.0) != []


def test_backtest_engine_executes_generated_order_on_next_bar() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 30.0, 40.0],
            "high": [11.0, 11.0, 11.0, 11.0, 11.0, 21.0, 31.0, 41.0],
            "low": [9.0, 9.0, 9.0, 9.0, 9.0, 19.0, 29.0, 39.0],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 30.0, 40.0],
            "volume": [1_000.0] * 8,
        },
        index=index,
    )
    engine = backtester.BacktestEngine(
        {"AAPL": frame},
        backtester.DefaultExecutionModel(),
        initial_cash=1_000.0,
    )

    result = engine.run(["AAPL"])

    assert not result.trades.empty
    first_trade = result.trades.iloc[0]
    assert first_trade["timestamp"] == index[7]
    assert first_trade["price"] == 40.0


def test_backtest_latency_fill_prices_at_release_bar() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [10.0, 20.0, 30.0],
            "volume": [1_000.0] * 3,
        },
        index=index,
    )
    engine = backtester.BacktestEngine(
        {"AAPL": frame},
        backtester.DefaultExecutionModel(latency=1),
        initial_cash=1_000.0,
    )
    generated = False

    def generate_once(symbol: str, close: float) -> list[backtester.Order]:
        nonlocal generated
        if generated:
            return []
        generated = True
        return [backtester.Order(symbol=symbol, qty=1, side="buy", price=close)]

    engine._generate_orders_for_bar = generate_once  # type: ignore[method-assign]

    result = engine.run(["AAPL"])

    assert not result.trades.empty
    first_trade = result.trades.iloc[0]
    assert first_trade["timestamp"] == index[2]
    assert first_trade["price"] == 30.0
