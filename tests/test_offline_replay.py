from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trading.tools.offline_replay import main


def _write_synthetic_bars(csv_path: Path, periods: int = 360) -> None:
    idx = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="min")
    x = np.linspace(0.0, 28.0, periods)
    drift = np.linspace(0.0, 1.5, periods)
    close = 100.0 + 1.4 * np.sin(x) + drift
    open_ = close + 0.03 * np.sin(x / 2.0)
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    volume = 10_000.0 + 250.0 * np.cos(x)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    frame.to_csv(csv_path, index=False)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_offline_replay_writes_summary_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "AAPL.csv"
    out_path = tmp_path / "summary.json"
    _write_synthetic_bars(csv_path)

    rc = main(
        [
            "--csv",
            str(csv_path),
            "--output-json",
            str(out_path),
            "--confidence-threshold",
            "0.10",
            "--entry-score-threshold",
            "0.05",
            "--min-hold-bars",
            "3",
            "--max-hold-bars",
            "40",
            "--take-profit-bps",
            "30",
            "--stop-loss-bps",
            "30",
            "--trailing-stop-bps",
            "20",
        ]
    )
    assert rc == 0
    payload = _load_json(out_path)
    assert payload["aggregate"]["symbols"] == 1
    assert payload["aggregate"]["total_bars"] == 360
    assert "total_trades" in payload["aggregate"]


def test_offline_replay_higher_min_hold_reduces_churn(tmp_path: Path) -> None:
    csv_path = tmp_path / "ABT.csv"
    out_fast = tmp_path / "fast.json"
    out_slow = tmp_path / "slow.json"
    _write_synthetic_bars(csv_path, periods=420)

    base_args = [
        "--csv",
        str(csv_path),
        "--confidence-threshold",
        "0.08",
        "--entry-score-threshold",
        "0.03",
        "--allow-shorts",
        "--max-hold-bars",
        "60",
        "--take-profit-bps",
        "25",
        "--stop-loss-bps",
        "25",
        "--trailing-stop-bps",
        "15",
    ]

    rc_fast = main(base_args + ["--min-hold-bars", "1", "--output-json", str(out_fast)])
    rc_slow = main(base_args + ["--min-hold-bars", "12", "--output-json", str(out_slow)])
    assert rc_fast == 0
    assert rc_slow == 0

    fast_trades = int(_load_json(out_fast)["aggregate"]["total_trades"])
    slow_trades = int(_load_json(out_slow)["aggregate"]["total_trades"])
    assert fast_trades > 0
    assert slow_trades <= fast_trades


def test_offline_replay_simulation_mode_is_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "QQQ.csv"
    out_first = tmp_path / "sim_first.json"
    out_second = tmp_path / "sim_second.json"
    _write_synthetic_bars(csv_path, periods=240)

    args = [
        "--csv",
        str(csv_path),
        "--simulation-mode",
        "--replay-seed",
        "123",
        "--confidence-threshold",
        "0.05",
        "--entry-score-threshold",
        "0.03",
    ]

    rc_first = main(args + ["--output-json", str(out_first)])
    rc_second = main(args + ["--output-json", str(out_second)])
    assert rc_first == 0
    assert rc_second == 0

    first = _load_json(out_first)
    second = _load_json(out_second)
    assert first["aggregate"]["simulation_mode"] is True
    assert second["aggregate"]["simulation_mode"] is True
    assert first["aggregate"]["replay_seed"] == 123
    assert second["aggregate"]["replay_seed"] == 123
    assert first["replay"]["events"] == second["replay"]["events"]
