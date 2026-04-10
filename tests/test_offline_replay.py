from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, cast

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


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_trending_bars(csv_path: Path, periods: int = 120) -> None:
    idx = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="min")
    close = np.linspace(100.0, 220.0, periods)
    open_ = close - 0.2
    high = close + 0.25
    low = open_ - 0.25
    volume = np.full(periods, 9_000.0)
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


def _write_duplicate_timestamp_bars(csv_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02T14:30:00Z",
                "2026-01-02T14:30:00Z",
                "2026-01-02T14:31:00Z",
                "2026-01-02T14:32:00Z",
            ],
            "open": [100.0, 100.2, 100.7, 101.1],
            "high": [100.3, 100.5, 101.0, 101.4],
            "low": [99.8, 100.0, 100.5, 100.9],
            "close": [100.1, 100.4, 100.9, 101.3],
            "volume": [5_000.0, 5_200.0, 5_100.0, 5_150.0],
        }
    )
    frame.to_csv(csv_path, index=False)


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


def test_offline_replay_policy_sensitivity_reports_per_knob_contributions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    aapl_path = tmp_path / "AAPL.csv"
    msft_path = tmp_path / "MSFT.csv"
    out_path = tmp_path / "policy_sensitivity.json"
    _write_synthetic_bars(aapl_path, periods=240)
    _write_trending_bars(msft_path, periods=240)

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    monkeypatch.setenv("AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE", "0.95")
    monkeypatch.setenv("AI_TRADING_EXEC_OPPORTUNITY_MIN_SYMBOLS", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR", "0.85")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS", "12.0")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_CONSTRAINT_WEIGHT", "1.20")
    monkeypatch.setenv("AI_TRADING_EXEC_REPLAY_QUALITY_WEIGHT", "0.25")
    monkeypatch.setenv("AI_TRADING_EXEC_REPLAY_QUALITY_MAX_RANK_UPLIFT_ABS", "8.0")
    monkeypatch.setenv("AI_TRADING_EXEC_REPLAY_QUALITY_MAX_RANK_UPLIFT_FRAC", "0.10")
    monkeypatch.setenv("AI_TRADING_EXEC_REPLAY_QUALITY_MAX_AGE_HOURS", "24.0")
    monkeypatch.setenv("AI_TRADING_EXEC_BANDIT_SCORE_WEIGHT", "0.40")
    monkeypatch.setenv("AI_TRADING_EXEC_BANDIT_MIN_SAMPLES", "10")
    monkeypatch.setenv("AI_TRADING_EXEC_BANDIT_SHADOW_ONLY", "0")

    rc = main(
        [
            "--data-dir",
            str(tmp_path),
            "--simulation-mode",
            "--policy-sensitivity-mode",
            "--replay-seed",
            "41",
            "--confidence-threshold",
            "0.0",
            "--entry-score-threshold",
            "0.0",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = _load_json(out_path)
    report = payload.get("policy_sensitivity", {})
    assert report.get("enabled") is True
    baseline = report.get("baseline", {})
    baseline_metrics = baseline.get("metrics", {})
    assert int(baseline_metrics.get("total_trades", 0)) >= 0
    variants = cast(list[dict[str, Any]], report.get("variants", []))
    assert len(variants) >= 8
    contributions = cast(list[dict[str, Any]], report.get("per_knob_contribution", []))
    assert len(contributions) >= 8
    names = {str(item.get("name", "")) for item in contributions}
    assert "opportunity_gate_disabled" in names
    assert "capture_floor_disabled" in names
    assert "replay_quality_disabled" in names
    assert "bandit_disabled" in names
    assert "bandit_live_enabled" in names
    assert "replay_quality_weight_0_10" in names
    assert "replay_quality_weight_0_25" in names
    assert "replay_quality_weight_0_40" in names
    assert any(
        abs(float(item.get("delta_expectancy_bps", 0.0))) > 1e-9
        or int(item.get("delta_total_trades", 0)) != 0
        for item in contributions
    )
    summary_table = cast(list[dict[str, Any]], report.get("summary_table", []))
    assert len(summary_table) == len(contributions)
    assert all(int(row.get("rank", 0)) == idx for idx, row in enumerate(summary_table, start=1))
    deltas = [float(row.get("delta_expectancy_bps", 0.0)) for row in summary_table]
    assert deltas == sorted(deltas, reverse=True)


def test_offline_replay_policy_sensitivity_honors_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    aapl_path = tmp_path / "AAPL.csv"
    msft_path = tmp_path / "MSFT.csv"
    out_path = tmp_path / "policy_env_file.json"
    env_file = tmp_path / "policy_replay.env"
    _write_synthetic_bars(aapl_path, periods=120)
    _write_trending_bars(msft_path, periods=120)
    env_file.write_text(
        "\n".join(
            [
                "AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE=0.82",
                "AI_TRADING_EXEC_OPPORTUNITY_MIN_SYMBOLS=3",
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR=0.31",
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS=4.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    rc = main(
        [
            "--data-dir",
            str(tmp_path),
            "--simulation-mode",
            "--policy-sensitivity-mode",
            "--env-file",
            str(env_file),
            "--confidence-threshold",
            "0.0",
            "--entry-score-threshold",
            "0.0",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = _load_json(out_path)
    baseline = payload["policy_sensitivity"]["baseline"]
    baseline_profile = baseline["profile"]
    assert baseline_profile["opportunity_top_quantile"] == pytest.approx(0.82, abs=1e-9)
    assert baseline_profile["opportunity_min_symbols"] == 3
    assert baseline_profile["expected_capture_fill_prob_floor"] == pytest.approx(0.31, abs=1e-9)
    assert baseline_profile["expected_capture_floor_bps"] == pytest.approx(4.5, abs=1e-9)


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


def test_offline_replay_simulation_mode_populates_markout_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "UPTREND.csv"
    out_path = tmp_path / "sim_metrics.json"
    _write_trending_bars(csv_path, periods=80)

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    rc = main(
        [
            "--csv",
            str(csv_path),
            "--simulation-mode",
            "--replay-seed",
            "17",
            "--confidence-threshold",
            "0.0",
            "--entry-score-threshold",
            "0.0",
            "--no-allow-shorts",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = _load_json(out_path)
    aggregate = payload["aggregate"]
    assert aggregate["simulation_mode"] is True
    assert aggregate["metrics_mode"] == "one_bar_markout"
    assert int(aggregate["markout_samples"]) > 0
    assert float(aggregate["expectancy_bps"]) != 0.0
    assert float(aggregate["net_pnl_bps"]) != 0.0


def test_offline_replay_duplicate_timestamps_do_not_raise_duplicate_intent_violation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "DUPTS.csv"
    out_path = tmp_path / "dup_ts.json"
    _write_duplicate_timestamp_bars(csv_path)

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    rc = main(
        [
            "--csv",
            str(csv_path),
            "--simulation-mode",
            "--replay-seed",
            "23",
            "--confidence-threshold",
            "0.0",
            "--entry-score-threshold",
            "0.0",
            "--no-allow-shorts",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = _load_json(out_path)
    violations = payload["aggregate"]["violations_by_code"]
    assert int(violations.get("duplicate_intent", 0)) == 0


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
