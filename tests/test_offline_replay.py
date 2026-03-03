from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

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


def test_offline_replay_simulation_mode_persists_intents_to_oms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("sqlalchemy")
    csv_path = tmp_path / "IWM.csv"
    out_path = tmp_path / "persist.json"
    oms_path = tmp_path / "oms_replay.db"
    _write_synthetic_bars(csv_path, periods=180)

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(oms_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")

    rc = main(
        [
            "--csv",
            str(csv_path),
            "--simulation-mode",
            "--persist-intents",
            "--replay-seed",
            "77",
            "--confidence-threshold",
            "0.05",
            "--entry-score-threshold",
            "0.03",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0

    payload = _load_json(out_path)
    summary = payload["aggregate"]["oms_persist_summary"]
    assert summary["persisted"] is True
    assert int(summary["created_intents"]) > 0
    assert int(summary["fill_events"]) > 0
    assert oms_path.exists()

    with sqlite3.connect(oms_path) as conn:
        intent_count = int(conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0])
        fill_count = int(conn.execute("SELECT COUNT(*) FROM intent_fills").fetchone()[0])
    assert intent_count >= int(summary["created_intents"])
    assert fill_count >= int(summary["fill_events"])


def test_offline_replay_persist_rerun_skips_terminal_existing_intents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("sqlalchemy")
    csv_path = tmp_path / "SPY.csv"
    first_out = tmp_path / "persist_first.json"
    second_out = tmp_path / "persist_second.json"
    oms_path = tmp_path / "oms_replay.db"
    _write_synthetic_bars(csv_path, periods=180)

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(oms_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")

    args = [
        "--csv",
        str(csv_path),
        "--simulation-mode",
        "--persist-intents",
        "--replay-seed",
        "77",
        "--confidence-threshold",
        "0.05",
        "--entry-score-threshold",
        "0.03",
    ]

    assert main(args + ["--output-json", str(first_out)]) == 0
    with sqlite3.connect(oms_path) as conn:
        first_intents = int(conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0])
        first_fills = int(conn.execute("SELECT COUNT(*) FROM intent_fills").fetchone()[0])

    assert main(args + ["--output-json", str(second_out)]) == 0
    second_payload = _load_json(second_out)
    second_summary = second_payload["aggregate"]["oms_persist_summary"]
    assert int(second_summary["created_intents"]) == 0
    assert int(second_summary["existing_intents"]) > 0
    assert int(second_summary["existing_terminal_intents_skipped"]) > 0
    assert int(second_summary["fill_events"]) == 0

    with sqlite3.connect(oms_path) as conn:
        second_intents = int(conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0])
        second_fills = int(conn.execute("SELECT COUNT(*) FROM intent_fills").fetchone()[0])

    assert second_intents == first_intents
    assert second_fills == first_fills


def test_offline_replay_rejects_directory_csv_input(tmp_path: Path) -> None:
    out_path = tmp_path / "bad.json"
    rc = main(
        [
            "--csv",
            str(tmp_path),
            "--simulation-mode",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 1
    assert not out_path.exists()
