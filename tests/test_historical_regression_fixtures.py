from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ai_trading.strategies import backtester
from ai_trading.tools.offline_replay import main as offline_replay_main


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _write_backtester_symbol(path: Path, *, offset: float, duplicate_ts: bool) -> None:
    closes = [
        100.0 + offset,
        101.0 + offset,
        102.0 + offset,
        103.0 + offset,
        102.0 + offset,
        101.0 + offset,
        100.0 + offset,
        99.0 + offset,
        100.0 + offset,
        102.0 + offset,
        104.0 + offset,
        103.0 + offset,
        101.0 + offset,
        100.0 + offset,
        101.0 + offset,
        103.0 + offset,
        105.0 + offset,
        104.0 + offset,
        102.0 + offset,
        101.0 + offset,
    ]
    ts_values = list(pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC"))
    if duplicate_ts:
        ts_values[7] = ts_values[6]
    frame = pd.DataFrame(
        {
            "event_time": ts_values,
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
        }
    )
    frame.to_csv(path, index=False)


def _write_replay_symbol(path: Path, *, trend: float, include_volume: bool, duplicate_ts: bool) -> None:
    periods = 240
    idx = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="min")
    if duplicate_ts:
        idx = idx.to_list()
        idx[40] = idx[39]
    x = np.linspace(0.0, 24.0, periods)
    close = 100.0 + trend + (1.2 * np.sin(x)) + np.linspace(0.0, 2.0, periods)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
        }
    )
    if include_volume:
        frame["volume"] = 12_000.0 + 250.0 * np.cos(x)
    frame.to_csv(path, index=False)


def _normalize_backtester_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized.pop("generated_at", None)
    normalized.pop("authority", None)
    normalized["config"]["data_dir"] = "__DATA_DIR__"
    normalized["artifacts"] = {
        key: Path(str(value)).name
        for key, value in normalized["artifacts"].items()
    }
    for report in normalized["inputs"]["symbols"].values():
        report["path"] = Path(str(report["path"])).name
        report.pop("timestamp_authoritative", None)
        report.pop("research_synthetic", None)
        report.pop("source_providers", None)
    normalized["symbols"] = sorted(normalized["symbols"], key=lambda item: str(item["symbol"]))
    for row in normalized["symbols"]:
        row["load_report"]["path"] = Path(str(row["load_report"]["path"])).name
        row["load_report"].pop("timestamp_authoritative", None)
        row["load_report"].pop("research_synthetic", None)
        row["load_report"].pop("source_providers", None)
    return cast(dict[str, Any], normalized)


def _normalize_offline_replay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized.pop("generated_at", None)
    normalized.pop("authority", None)
    artifacts = normalized.get("artifacts", {})
    if isinstance(artifacts, dict):
        normalized["artifacts"] = {
            key: Path(str(value)).name
            for key, value in artifacts.items()
        }
    for report in normalized["inputs"]["symbols"].values():
        report["path"] = Path(str(report["path"])).name
        report.pop("timestamp_authoritative", None)
        report.pop("research_synthetic", None)
        report.pop("source_providers", None)
    normalized["symbols"] = [
        {
            key: value
            for key, value in item.items()
            if key != "trades_detail"
        }
        for item in sorted(normalized["symbols"], key=lambda row: str(row["symbol"]))
    ]
    return cast(dict[str, Any], normalized)


def test_backtester_large_run_matches_golden_fixture(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_backtester_symbol(data_dir / "AAPL.csv", offset=0.0, duplicate_ts=False)
    _write_backtester_symbol(data_dir / "MSFT.csv", offset=12.0, duplicate_ts=True)

    backtester.main(
        [
            "--symbols",
            "AAPL",
            "MSFT",
            "--data-dir",
            str(data_dir),
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-31",
            "--timestamp-col",
            "event_time",
            "--output-dir",
            str(out_dir),
            "--commission",
            "0.01",
            "--slippage-pips",
            "0.02",
            "--latency-bars",
            "1",
        ]
    )

    payload = json.loads((out_dir / "backtest_summary.json").read_text(encoding="utf-8"))
    normalized = _normalize_backtester_payload(payload)
    expected = json.loads((FIXTURE_DIR / "backtester_large_run_expected.json").read_text(encoding="utf-8"))
    assert normalized == expected


def test_offline_replay_large_run_matches_golden_fixture(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_path = tmp_path / "offline_replay.json"
    data_dir.mkdir()
    _write_replay_symbol(data_dir / "AAPL.csv", trend=0.0, include_volume=True, duplicate_ts=False)
    _write_replay_symbol(data_dir / "MSFT.csv", trend=8.0, include_volume=False, duplicate_ts=True)

    rc = offline_replay_main(
        [
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL,MSFT",
            "--confidence-threshold",
            "0.08",
            "--entry-score-threshold",
            "0.03",
            "--min-hold-bars",
            "3",
            "--max-hold-bars",
            "45",
            "--stop-loss-bps",
            "25",
            "--take-profit-bps",
            "35",
            "--trailing-stop-bps",
            "15",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    normalized = _normalize_offline_replay_payload(payload)
    expected = json.loads((FIXTURE_DIR / "offline_replay_large_run_expected.json").read_text(encoding="utf-8"))
    assert normalized == expected
