from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

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
                "quote_age_ms": 200.0,
            },
            "provider": {
                "active": "alpaca-iex",
                "primary": "alpaca-iex",
                "status": "healthy",
                "using_backup": False,
            },
            "cost": {"spread_bps": 4.0, "quote_age_ms": 200.0},
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
                "quote_age_ms": 300.0,
            },
            "provider": {
                "active": "alpaca-iex",
                "primary": "alpaca-iex",
                "status": "healthy",
                "using_backup": False,
            },
            "cost": {"spread_bps": 5.0, "quote_age_ms": 300.0},
            "skew": {"breached": True},
        },
        {
            "ts": "2026-01-02T15:00:00+00:00",
            "mode": "ml_signal_shadow",
            "symbol": "AAPL",
            "champion_would_trade": True,
            "challenger_would_trade": False,
            "champion_probability": 0.55,
            "challenger_probability": 0.52,
            "probability_delta": 0.03,
            "market": {
                "bar_timestamp": "2026-01-02T00:00:00+00:00",
                "entry_close": 100.0,
                "spread_bps": 8.0,
                "quote_age_ms": 400.0,
            },
            "provider": {
                "active": "yahoo",
                "primary": "alpaca-iex",
                "status": "degraded",
                "using_backup": True,
            },
            "cost": {"spread_bps": 8.0, "quote_age_ms": 400.0},
            "skew": {"breached": False},
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
            horizon_bars_list="1,3",
            fee_bps=0.0,
            slippage_bps=0.0,
            frame_filter="all",
            provider_filter="all",
            since="",
            min_informational_rows=2,
            min_review_rows=3,
        )
    )

    assert output_json.is_file()
    persisted = cast(dict[str, Any], json.loads(output_json.read_text(encoding="utf-8")))
    assert persisted == report
    assert report["raw_rows"] == 3
    assert report["filtered_rows"] == 3
    assert report["sample_gate"]["status"] == "review_eligible"
    assert report["raw_frame_counts"] == {"daily": 1, "minute": 2}
    assert report["provider_summary"]["healthy_primary_rows"] == 2
    assert report["provider_summary"]["using_backup_rows"] == 1
    assert report["cost_observation_summary"]["mean_quote_age_ms"] == 300.0
    decisions = report["decision_summary"]
    assert decisions["rows"] == 3
    assert decisions["agreement_count"] == 1
    assert decisions["challenger_only_count"] == 1
    assert decisions["mean_probability_delta"] == 0.13
    assert decisions["mean_spread_bps"] == pytest.approx(5.666666666666667)
    assert decisions["skew_breach_count"] == 1
    markout = report["markout_summary"]
    assert markout["challenger_samples"] == 2
    assert markout["shadow_only_samples"] == 1
    assert markout["challenger_mean_net_markout_bps"] > 0.0
    assert markout["shadow_only_mean_net_markout_bps"] > 0.0
    assert sorted(report["markout_summaries"]) == ["1", "3"]
    assert report["markout_summaries"]["3"]["horizon_bars"] == 3


def test_build_shadow_report_filters_to_minute_rows(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "ml_shadow.jsonl"
    output_json = tmp_path / "shadow_report_minute.json"
    _write_shadow_jsonl(input_jsonl)

    report = build_shadow_report(
        argparse.Namespace(
            input_jsonl=input_jsonl,
            output_json=output_json,
            data_dir=None,
            timestamp_col="timestamp",
            horizon_bars=1,
            horizon_bars_list="",
            fee_bps=0.0,
            slippage_bps=0.0,
            frame_filter="minute",
            provider_filter="all",
            since="",
            min_informational_rows=100,
            min_review_rows=500,
        )
    )

    assert report["frame_filter"] == "minute"
    assert report["raw_rows"] == 3
    assert report["filtered_rows"] == 2
    decisions = report["decision_summary"]
    assert decisions["rows"] == 2
    assert decisions["champion_only_count"] == 0
    assert decisions["challenger_only_count"] == 1


def test_build_shadow_report_filters_by_since_and_provider(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "ml_shadow.jsonl"
    output_json = tmp_path / "shadow_report_filtered.json"
    _write_shadow_jsonl(input_jsonl)

    report = build_shadow_report(
        argparse.Namespace(
            input_jsonl=input_jsonl,
            output_json=output_json,
            data_dir=None,
            timestamp_col="timestamp",
            horizon_bars=1,
            horizon_bars_list="",
            fee_bps=0.0,
            slippage_bps=0.0,
            frame_filter="all",
            provider_filter="healthy-primary",
            since="2026-01-02T14:31:00Z",
            min_informational_rows=2,
            min_review_rows=5,
        )
    )

    assert report["filtered_rows"] == 1
    assert report["filter_counts"] == {
        "raw_rows": 3,
        "after_since": 2,
        "after_frame_filter": 2,
        "after_provider_filter": 1,
    }
    assert report["sample_gate"]["status"] == "insufficient"
    decisions = report["decision_summary"]
    assert decisions["rows"] == 1
    assert decisions["both_trade_count"] == 1
