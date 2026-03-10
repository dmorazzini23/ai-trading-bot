from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_trading import meta_learning
from ai_trading.tools.refresh_meta_model import main


def test_refresh_meta_model_tool_retrains_and_writes_model(tmp_path: Path) -> None:
    trade_log = tmp_path / "trades.csv"
    pd.DataFrame(
        {
            "entry_price": [100.0, 101.0, 102.0],
            "exit_price": [101.0, 102.5, 101.5],
            "signal_tags": ["momentum", "trend", "momentum+trend"],
            "side": ["buy", "buy", "sell"],
        }
    ).to_csv(trade_log, index=False)

    model_path = tmp_path / "meta_model.pkl"
    history_path = tmp_path / "meta_retrain_history.pkl"

    rc = main(
        [
            "--trade-log-path",
            str(trade_log),
            "--model-path",
            str(model_path),
            "--history-path",
            str(history_path),
            "--min-samples",
            "1",
        ]
    )
    assert rc == 0
    assert model_path.exists()
    assert history_path.exists()
    history = meta_learning.load_model_checkpoint(str(history_path))
    assert isinstance(history, list)
    assert history


def test_refresh_meta_model_tool_retrains_from_fill_derived_history(
    tmp_path: Path,
) -> None:
    fill_history = tmp_path / "tca_records.jsonl"
    rows = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 5,
            "fill_price": 100.0,
            "signal_tags": "momentum",
            "strategy": "swing",
            "ts": "2026-03-10T20:00:00+00:00",
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 5,
            "fill_price": 101.0,
            "signal_tags": "mean_revert",
            "strategy": "swing",
            "ts": "2026-03-10T20:05:00+00:00",
        },
    ]
    fill_history.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    model_path = tmp_path / "meta_model.pkl"
    history_path = tmp_path / "meta_retrain_history.pkl"
    output_json = tmp_path / "refresh_report.json"

    rc = main(
        [
            "--trade-log-path",
            str(fill_history),
            "--model-path",
            str(model_path),
            "--history-path",
            str(history_path),
            "--min-samples",
            "1",
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    assert model_path.exists()
    assert history_path.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["source_metadata"]["source_mode"] == "fill_derived"
    assert report["quality_report"]["valid_price_rows"] >= 1
    # Materialized dataset should be a CSV, not the original jsonl fill file.
    assert str(report["trade_log_path"]).endswith(".csv")
    assert Path(report["trade_log_path"]).exists()
