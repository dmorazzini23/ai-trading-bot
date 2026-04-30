from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ai_trading.tools.ml_shadow_report import build_shadow_report


def _write_bars(path: Path) -> None:
    idx = pd.date_range("2026-01-02T14:30:00Z", periods=5, freq="min")
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000.0] * 5,
        }
    ).to_csv(path, index=False)


def _write_shadow_jsonl(path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {
            "ts": "2026-01-02T14:30:00+00:00",
            "mode": "ml_signal_shadow",
            "symbol": "AAPL",
            "champion_would_trade": False,
            "challenger_would_trade": True,
            "champion_probability": 0.40,
            "challenger_probability": 0.72,
            "probability_delta": 0.32,
            "market": {
                "bar_timestamp": "2026-01-02T14:30:00+00:00",
                "entry_close": 100.0,
                "spread_bps": 4.0,
            },
            "skew": {"breached": False},
        },
        {
            "ts": "2026-01-02T14:31:00+00:00",
            "mode": "ml_signal_shadow",
            "symbol": "AAPL",
            "champion_would_trade": True,
            "challenger_would_trade": True,
            "champion_probability": 0.80,
            "challenger_probability": 0.76,
            "probability_delta": 0.04,
            "market": {
                "bar_timestamp": "2026-01-02T14:31:00+00:00",
                "entry_close": 101.0,
                "spread_bps": 5.0,
            },
            "skew": {"breached": True},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_shadow_report_summarizes_decisions_and_markout(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    data_dir.mkdir()
    _write_bars(data_dir / "AAPL.csv")
    input_jsonl = tmp_path / "ml_shadow.jsonl"
    output_json = tmp_path / "shadow_report.json"
    _write_shadow_jsonl(input_jsonl)

    report = build_shadow_report(
        argparse.Namespace(
            input_jsonl=input_jsonl,
            output_json=output_json,
            data_dir=data_dir,
            timestamp_col="timestamp",
            horizon_bars=1,
            fee_bps=0.0,
            slippage_bps=0.0,
        )
    )

    assert output_json.is_file()
    persisted = cast(dict[str, Any], json.loads(output_json.read_text(encoding="utf-8")))
    assert persisted == report
    decisions = report["decision_summary"]
    assert decisions["rows"] == 2
    assert decisions["agreement_count"] == 1
    assert decisions["challenger_only_count"] == 1
    assert decisions["mean_probability_delta"] == 0.18
    assert decisions["mean_spread_bps"] == 4.5
    assert decisions["skew_breach_count"] == 1
    markout = report["markout_summary"]
    assert markout["challenger_samples"] == 2
    assert markout["shadow_only_samples"] == 1
    assert markout["challenger_mean_net_markout_bps"] > 0.0
    assert markout["shadow_only_mean_net_markout_bps"] > 0.0
