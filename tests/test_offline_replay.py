from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.execution.simulated_broker import SimulatedBroker
from ai_trading.replay.event_loop import ReplayEventLoop
from ai_trading.tools import offline_replay as replay_tool
from ai_trading.tools.offline_replay import (
    _accepted_candidate_row,
    _resolve_markout_veto_config,
    main,
)


@pytest.fixture(autouse=True)
def _disable_default_runtime_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", "")


class _ConstantEdgeModel:
    def __init__(
        self,
        *,
        p_edge: float,
        orientation: str = "direct",
        penalties: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._p_edge = float(np.clip(p_edge, 0.0, 1.0))
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.feature_names_in_ = np.asarray(
            [
                "rsi",
                "macd",
                "atr",
                "vwap",
                "sma_50",
                "sma_200",
                "signal",
                "atr_pct",
                "vwap_distance",
                "sma_spread",
                "macd_signal_gap",
                "rsi_centered",
            ],
            dtype=object,
        )
        self.edge_score_orientation_ = str(orientation)
        self.edge_negative_symbol_penalties_ = penalties or {}

    def predict_proba(self, X: Any) -> np.ndarray:
        rows = int(np.asarray(X).shape[0])
        edge: np.ndarray = np.full(rows, self._p_edge, dtype=float)
        return cast(np.ndarray, np.column_stack((1.0 - edge, edge)))

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return cast(np.ndarray, (probs >= 0.5).astype(int))


def test_replay_event_loop_allows_long_short_netting_reductions() -> None:
    bars = [{"ts": "2026-05-05T14:30:00Z", "symbol": "AAPL", "close": 100.0}]

    for initial_qty, side, expected_qty in ((10.0, "sell", 5.0), (-10.0, "buy", -5.0)):
        loop = ReplayEventLoop(
            strategy=lambda _bar, side=side: {
                "symbol": "AAPL",
                "side": side,
                "qty": 5.0,
                "type": "limit",
                "price": 100.0,
            },
            broker=SimulatedBroker(
                seed=7,
                fill_probability=1.0,
                partial_fill_probability=0.0,
                min_fill_delay_ms=0,
                max_fill_delay_ms=0,
            ),
            max_symbol_notional=800.0,
            max_gross_notional=800.0,
            initial_positions={"AAPL": initial_qty},
            clip_intents_to_caps=True,
        )

        payload = loop.run(bars)

        assert payload["violations"] == []
        assert payload["cap_adjustments"] == []
        assert payload["orders"][0]["qty"] == pytest.approx(5.0)
        assert payload["positions"]["AAPL"] == pytest.approx(expected_qty)


def test_replay_markout_metrics_subtract_live_round_trip_costs() -> None:
    metrics = replay_tool._summarize_markout_fill_metrics(
        fill_events=[
            {
                "event_type": "fill",
                "client_order_id": "order-1",
                "symbol": "AAPL",
                "side": "buy",
                "fill_qty": 2.0,
                "fill_price": 100.0,
            }
        ],
        order_context_by_client_id={
            "order-1": {
                "markout_price": 101.0,
                "entry_slippage_bps": 4.0,
                "exit_slippage_bps": 6.0,
            }
        },
        fee_bps=1.0,
    )

    assert metrics["expectancy_bps"] == pytest.approx(88.0)
    assert metrics["mean_round_trip_cost_bps"] == pytest.approx(12.0)


def _write_model(
    path: Path,
    *,
    p_edge: float,
    orientation: str = "direct",
    penalties: dict[str, dict[str, float]] | None = None,
) -> None:
    import joblib

    model = _ConstantEdgeModel(
        p_edge=p_edge,
        orientation=orientation,
        penalties=penalties,
    )
    joblib.dump(model, path)
    write_artifact_manifest(
        model_path=path,
        model_version="offline-replay-test-model-v1",
        metadata={"source": "tests.test_offline_replay"},
    )


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


def _write_downtrending_bars(csv_path: Path, periods: int = 120) -> None:
    idx = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="min")
    close = np.linspace(220.0, 100.0, periods)
    open_ = close + 0.2
    high = open_ + 0.25
    low = close - 0.25
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


def _write_live_cost_replay_artifact(path: Path, *, slippage_bps: float) -> None:
    rows: list[dict[str, object]] = []
    for session_regime in ("opening", "midday", "closing"):
        for side in ("buy", "sell", "sell_short"):
            rows.append(
                {
                    "symbol": "AAPL",
                    "side": side,
                    "session_regime": session_regime,
                    "sample_count": 10,
                    "sufficient_samples": True,
                    "p90_adverse_slippage_bps": float(slippage_bps),
                    "mean_slippage_bps": float(slippage_bps) / 2.0,
                }
            )
    path.write_text(
        json.dumps(
            {
                "artifact_type": "live_cost_model",
                "generated_at": "2026-01-02T21:00:00Z",
                "status": {"available": True, "status": "ready"},
                "by_symbol_side_session": rows,
            }
        ),
        encoding="utf-8",
    )


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
    assert payload["schema_version"] == "1.0.0"
    assert payload["artifact_type"] == "offline_replay_summary"
    assert payload["aggregate"]["symbols"] == 1
    assert payload["aggregate"]["total_bars"] == 360
    assert payload["artifacts"]["output_json"] == str(out_path)
    assert payload["inputs"]["symbols"]["AAPL"]["rows_after_cleanup"] == 360
    assert payload["authority"]["timestamp_authoritative"] is True
    assert payload["authority"]["research_synthetic"] is False
    assert "total_trades" in payload["aggregate"]


def test_offline_replay_rejects_non_timestamped_csv_by_default(tmp_path: Path) -> None:
    csv_path = tmp_path / "AAPL.csv"
    pd.DataFrame(
        {
            "seq": [0, 1, 2, 3],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10, 10, 10, 10],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="requires timestamp-authoritative bars"):
        replay_tool.run_replay(["--csv", str(csv_path)])


def test_offline_replay_live_cost_model_updates_slippage_assumptions(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    base_out = tmp_path / "base.json"
    live_cost_out = tmp_path / "live_cost.json"
    live_cost_model = tmp_path / "live_cost_model_latest.json"
    _write_synthetic_bars(csv_path)
    _write_live_cost_replay_artifact(live_cost_model, slippage_bps=25.0)

    base_args = [
        "--csv",
        str(csv_path),
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
        "--fee-bps",
        "0",
        "--slippage-bps",
        "0",
    ]

    assert main([*base_args, "--output-json", str(base_out)]) == 0
    assert (
        main(
            [
                *base_args,
                "--live-cost-model-json",
                str(live_cost_model),
                "--output-json",
                str(live_cost_out),
            ]
        )
        == 0
    )

    base_payload = _load_json(base_out)
    live_cost_payload = _load_json(live_cost_out)
    assert base_payload["aggregate"]["total_trades"] > 0
    assert live_cost_payload["aggregate"]["total_trades"] > 0
    assert (
        live_cost_payload["aggregate"]["net_pnl_bps"]
        < base_payload["aggregate"]["net_pnl_bps"]
    )
    config = live_cost_payload["aggregate"]["config"]["live_cost_model"]
    assert config["enabled"] is True
    assert config["path"] == str(live_cost_model)
    assert config["bucket_count"] == 9


def test_offline_replay_confidence_sizing_policy_records_scaled_trades(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    flat_out = tmp_path / "flat.json"
    sized_out = tmp_path / "sized.json"
    _write_synthetic_bars(csv_path)
    base_args = [
        "--csv",
        str(csv_path),
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

    assert main([*base_args, "--output-json", str(flat_out)]) == 0
    assert (
        main(
            [
                *base_args,
                "--sizing-policy",
                "confidence",
                "--sizing-min-scale",
                "0.50",
                "--sizing-max-scale",
                "2.0",
                "--output-json",
                str(sized_out),
            ]
        )
        == 0
    )

    flat_payload = _load_json(flat_out)
    sized_payload = _load_json(sized_out)
    assert sized_payload["aggregate"]["config"]["sizing_policy"] == "confidence"
    assert sized_payload["aggregate"]["avg_size_multiplier"] != pytest.approx(1.0)
    assert sized_payload["aggregate"]["net_pnl_bps"] != pytest.approx(
        flat_payload["aggregate"]["net_pnl_bps"]
    )
    first_trade = sized_payload["symbols"][0]["trades_detail"][0]
    assert first_trade["size_multiplier"] > 0.0
    assert first_trade["sizing"]["policy"] == "confidence"
    assert "raw_pnl_bps" in first_trade


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
    assert first["schema_version"] == "1.0.0"
    assert first["aggregate"]["simulation_mode"] is True
    assert second["aggregate"]["simulation_mode"] is True
    assert first["aggregate"]["replay_seed"] == 123
    assert second["aggregate"]["replay_seed"] == 123
    assert first["replay"]["events"] == second["replay"]["events"]
    quality = first["aggregate"]["candidate_quality"]
    assert quality["overall"]["candidates"] == first["aggregate"]["accepted_candidate_count"]
    assert quality["by_side"]
    assert quality["by_score_bucket"]
    assert quality["by_confidence_bucket"]
    assert quality["by_rank_bucket"]
    assert "cap_adjustments" in quality


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
    baseline_diag = cast(dict[str, Any], baseline.get("policy_diagnostics", {}))
    rejected_by_reason = cast(dict[str, Any], baseline_diag.get("rejected_by_reason", {}))
    gate_bind_rates = cast(dict[str, Any], baseline_diag.get("gate_bind_rates", {}))
    gate_bind_ranked = cast(list[dict[str, Any]], baseline_diag.get("gate_bind_ranked", []))
    assert int(baseline_diag.get("candidates", 0)) >= 0
    assert int(baseline_diag.get("accepted", 0)) >= 0
    assert int(baseline_diag.get("rejected_total", 0)) == sum(
        int(value) for value in rejected_by_reason.values()
    )
    if int(baseline_diag.get("candidates", 0)) > 0:
        assert float(baseline_diag.get("rejection_rate", 0.0)) == pytest.approx(
            int(baseline_diag.get("rejected_total", 0))
            / int(baseline_diag.get("candidates", 1)),
            abs=1e-9,
        )
    for reason, count_any in rejected_by_reason.items():
        count = int(count_any)
        detail = cast(dict[str, Any], gate_bind_rates.get(str(reason), {}))
        assert int(detail.get("count", -1)) == count
        if int(baseline_diag.get("candidates", 0)) > 0:
            assert float(detail.get("bind_rate_of_candidates", -1.0)) == pytest.approx(
                count / int(baseline_diag.get("candidates", 1)),
                abs=1e-9,
            )
        if int(baseline_diag.get("rejected_total", 0)) > 0:
            assert float(detail.get("share_of_rejections", -1.0)) == pytest.approx(
                count / int(baseline_diag.get("rejected_total", 1)),
                abs=1e-9,
            )
    ranked_counts = [int(item.get("count", 0)) for item in gate_bind_ranked]
    assert ranked_counts == sorted(ranked_counts, reverse=True)
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


def test_offline_replay_env_vars_take_precedence_over_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "OVR.csv"
    out_path = tmp_path / "override.json"
    env_file = tmp_path / "override.env"
    _write_synthetic_bars(csv_path, periods=140)
    env_file.write_text(
        "\n".join(
            [
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR=0.95",
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS=1000000.0",
            ]
        ),
        encoding="utf-8",
    )

    # Process env overrides should win over values in --env-file.
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR", "0.05")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS", "-1000000.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    rc = main(
        [
            "--csv",
            str(csv_path),
            "--simulation-mode",
            "--apply-policy-controls",
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
    policy_diag = cast(dict[str, Any], payload["aggregate"].get("policy_diagnostics", {}))
    profile = cast(dict[str, Any], policy_diag.get("profile", {}))
    assert profile["expected_capture_fill_prob_floor"] == pytest.approx(0.05, abs=1e-9)
    assert profile["expected_capture_floor_bps"] == pytest.approx(-1000000.0, abs=1e-9)


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


def test_offline_replay_apply_policy_controls_changes_simulation_outcome(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "CTRL.csv"
    plain_out = tmp_path / "plain.json"
    policy_out = tmp_path / "policy.json"
    _write_synthetic_bars(csv_path, periods=160)

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    # Force policy controls to reject nearly all candidate orders.
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS", "1000000.0")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR", "0.95")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_CONSTRAINT_WEIGHT", "1.2")

    base_args = [
        "--csv",
        str(csv_path),
        "--simulation-mode",
        "--replay-seed",
        "19",
        "--confidence-threshold",
        "0.0",
        "--entry-score-threshold",
        "0.0",
        "--output-json",
    ]
    rc_plain = main(base_args + [str(plain_out)])
    rc_policy = main(base_args + [str(policy_out), "--apply-policy-controls"])
    assert rc_plain == 0
    assert rc_policy == 0

    plain_payload = _load_json(plain_out)
    policy_payload = _load_json(policy_out)
    plain_agg = plain_payload["aggregate"]
    policy_agg = policy_payload["aggregate"]

    assert plain_agg["simulation_mode"] is True
    assert policy_agg["simulation_mode"] is True
    assert plain_agg["policy_controls_applied"] is False
    assert policy_agg["policy_controls_applied"] is True
    assert policy_agg["policy_mode"] == "ranker_controls"
    assert int(plain_agg["total_trades"]) > 0
    assert int(policy_agg["total_trades"]) < int(plain_agg["total_trades"])

    policy_diag = cast(dict[str, Any], policy_agg.get("policy_diagnostics", {}))
    rejected = cast(dict[str, Any], policy_diag.get("rejected_by_reason", {}))
    assert (
        int(rejected.get("expected_capture_floor", 0))
        + int(rejected.get("fill_prob_floor", 0))
    ) > 0


def test_offline_replay_opportunity_openings_only_skips_quantile_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    low_idx = pd.date_range("2026-01-02 14:30:00+00:00", periods=3, freq="min")
    high_idx = pd.date_range("2026-01-02 14:31:00+00:00", periods=2, freq="min")
    for symbol, idx, close in (
        ("LOW", low_idx, [100.0, 100.2, 100.4]),
        ("HIGH", high_idx, [101.0, 101.2]),
    ):
        pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": [value + 0.1 for value in close],
                "low": [value - 0.1 for value in close],
                "close": close,
                "volume": [10_000.0] * len(close),
            }
        ).to_csv(data_dir / f"{symbol}.csv", index=False)

    def _constant_signal(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return (
            pd.Series([1.0] * len(frame), index=frame.index),
            pd.Series([1.0] * len(frame), index=frame.index),
        )

    def _rank_low_second_after_first_open(bars: list[dict[str, Any]]) -> None:
        for bar in bars:
            symbol = str(bar["symbol"])
            ts = pd.Timestamp(str(bar["ts"]))
            if symbol == "LOW" and ts.minute == 30:
                rank_index = 0
                group_size = 1
            elif symbol == "LOW":
                rank_index = 1
                group_size = 2
            else:
                rank_index = 0
                group_size = 2
            bar["policy_fill_prob_proxy"] = 1.0
            bar["policy_expected_capture_proxy_bps"] = 10.0
            bar["policy_replay_quality_proxy_bps"] = 0.0
            bar["policy_bandit_proxy_bps"] = 0.0
            bar["policy_bandit_samples"] = 0
            bar["policy_bar_age_hours"] = 0.0
            bar["policy_rank_index"] = rank_index
            bar["policy_group_size"] = group_size

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE", "0.99")
    monkeypatch.setenv("AI_TRADING_EXEC_OPPORTUNITY_MIN_SYMBOLS", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR", "0.1")
    monkeypatch.setenv("AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS", "0.0")
    monkeypatch.setattr(replay_tool, "_compute_signal", _constant_signal)
    monkeypatch.setattr(replay_tool, "_attach_policy_context", _rank_low_second_after_first_open)

    out_path = tmp_path / "openings_only.json"
    assert main(
        [
            "--data-dir",
            str(data_dir),
            "--simulation-mode",
            "--apply-policy-controls",
            "--confidence-threshold",
            "0.0",
            "--entry-score-threshold",
            "0.01",
            "--output-json",
            str(out_path),
        ]
    ) == 0

    aggregate = _load_json(out_path)["aggregate"]
    diagnostics = aggregate["policy_diagnostics"]
    assert diagnostics["opportunity_openings_only_skipped"] == 3
    assert int(aggregate["accepted_candidate_count"]) == 5


def test_offline_replay_exports_accepted_candidate_components(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "EXPORT.csv"
    out_path = tmp_path / "export.json"
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
            "--export-accepted-candidates",
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
    artifacts = cast(dict[str, Any], payload.get("artifacts", {}))
    csv_artifact = Path(str(artifacts["accepted_candidates_csv"]))
    jsonl_artifact = Path(str(artifacts["accepted_candidates_jsonl"]))
    assert csv_artifact.exists()
    assert jsonl_artifact.exists()

    rows = list(csv.DictReader(csv_artifact.open("r", encoding="utf-8")))
    assert rows
    first = rows[0]
    assert first["symbol"] == "EXPORT"
    assert first["side"] == "buy"
    assert first["score"] != ""
    assert first["confidence"] != ""
    assert first["markout_bps"] != ""
    assert first["direction_correct"] in {"True", "False"}
    assert int(payload["aggregate"]["accepted_candidate_count"]) == len(rows)
    assert jsonl_artifact.read_text(encoding="utf-8").strip()


def test_offline_replay_markout_veto_shadow_and_enforce(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "VETO.csv"
    model_path = tmp_path / "veto_model.joblib"
    shadow_out = tmp_path / "shadow.json"
    enforce_out = tmp_path / "enforce.json"
    _write_downtrending_bars(csv_path, periods=80)
    _write_model(model_path, p_edge=0.9, orientation="direct")

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    base_args = [
        "--csv",
        str(csv_path),
        "--simulation-mode",
        "--use-model-score",
        "--model-path",
        str(model_path),
        "--confidence-threshold",
        "0.0",
        "--entry-score-threshold",
        "0.0",
        "--no-allow-shorts",
        "--markout-veto-lookback",
        "6",
        "--markout-veto-min-samples",
        "3",
        "--markout-veto-min-mean-bps",
        "0.0",
        "--markout-veto-max-wrong-way-rate",
        "0.50",
    ]
    assert main(base_args + ["--markout-veto-mode", "shadow", "--output-json", str(shadow_out)]) == 0
    assert main(base_args + ["--markout-veto-mode", "enforce", "--output-json", str(enforce_out)]) == 0

    shadow = _load_json(shadow_out)["aggregate"]
    enforced = _load_json(enforce_out)["aggregate"]
    assert shadow["markout_veto"]["mode"] == "shadow"
    assert enforced["markout_veto"]["mode"] == "enforce"
    assert int(shadow["markout_veto"]["shadow_flagged"]) > 0
    assert int(enforced["markout_veto"]["rejected"]) > 0
    assert int(enforced["total_trades"]) < int(shadow["total_trades"])
    shadow_quality = shadow["markout_veto"]["shadow_quality"]
    assert shadow_quality["flagged"]["candidates"] > 0
    assert shadow_quality["unflagged"]["candidates"] > 0
    assert shadow_quality["recommendation"] in {"enforce_candidate", "shadow_only"}


def test_markout_veto_config_preserves_zero_wrong_way_rate() -> None:
    args = type(
        "Args",
        (),
        {
            "markout_veto_mode": "enforce",
            "markout_veto_lookback": 20,
            "markout_veto_min_samples": 5,
            "markout_veto_min_mean_bps": 0.0,
            "markout_veto_max_wrong_way_rate": 0.0,
        },
    )()

    cfg = _resolve_markout_veto_config(args)

    assert cfg is not None
    assert cfg.max_wrong_way_rate == 0.0


def test_accepted_candidate_row_keeps_unknown_direction_blank() -> None:
    row = _accepted_candidate_row(
        context={
            "side": "buy",
            "submit_price": 100.0,
            "markout_price": None,
        },
        fill_event=None,
        cfg=cast(Any, type("Cfg", (), {"fee_bps": 0.0})()),
    )

    assert row["markout_bps"] is None
    assert row["direction_correct"] is None


def test_offline_replay_apply_policy_controls_requires_simulation_mode(tmp_path: Path) -> None:
    csv_path = tmp_path / "REQ.csv"
    out_path = tmp_path / "req.json"
    _write_synthetic_bars(csv_path, periods=80)

    rc = main(
        [
            "--csv",
            str(csv_path),
            "--apply-policy-controls",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 1


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


def test_offline_replay_model_scoring_sanitizes_duplicate_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    csv_path = tmp_path / "DUP_MODEL.csv"
    model_path = tmp_path / "dup_model.joblib"
    out_path = tmp_path / "dup_model_replay.json"
    _write_duplicate_timestamp_bars(csv_path)
    _write_model(model_path, p_edge=0.9, orientation="direct")

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    with caplog.at_level(logging.WARNING):
        rc = main(
            [
                "--csv",
                str(csv_path),
                "--simulation-mode",
                "--replay-seed",
                "29",
                "--confidence-threshold",
                "0.0",
                "--entry-score-threshold",
                "0.0",
                "--no-allow-shorts",
                "--use-model-score",
                "--model-path",
                str(model_path),
                "--output-json",
                str(out_path),
            ]
        )
    assert rc == 0
    payload = _load_json(out_path)
    assert payload["aggregate"]["model_score"]["enabled"] is True
    assert int(payload["aggregate"]["total_trades"]) > 0
    assert not any(record.msg == "OFFLINE_REPLAY_MODEL_SCORING_FAILED" for record in caplog.records)


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


def test_offline_replay_model_scoring_respects_inverse_orientation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "ORI.csv"
    direct_model = tmp_path / "direct_model.joblib"
    inverse_model = tmp_path / "inverse_model.joblib"
    out_direct = tmp_path / "direct.json"
    out_inverse = tmp_path / "inverse.json"
    _write_synthetic_bars(csv_path, periods=120)
    _write_model(direct_model, p_edge=0.9, orientation="direct")
    _write_model(inverse_model, p_edge=0.9, orientation="inverse")

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    args = [
        "--csv",
        str(csv_path),
        "--simulation-mode",
        "--no-allow-shorts",
        "--confidence-threshold",
        "0.0",
        "--entry-score-threshold",
        "0.0",
    ]
    assert main(args + ["--model-path", str(direct_model), "--output-json", str(out_direct)]) == 0
    assert main(args + ["--model-path", str(inverse_model), "--output-json", str(out_inverse)]) == 0

    direct_payload = _load_json(out_direct)
    inverse_payload = _load_json(out_inverse)
    assert int(direct_payload["aggregate"]["total_trades"]) > 0
    assert int(inverse_payload["aggregate"]["total_trades"]) == 0
    assert direct_payload["aggregate"]["model_score"]["enabled"] is True
    assert inverse_payload["aggregate"]["model_score"]["orientation"] == "inverse"


def test_offline_replay_long_only_model_probabilities_do_not_create_shorts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "LONGONLY.csv"
    model_path = tmp_path / "long_only_model.joblib"
    out_path = tmp_path / "long_only.json"
    _write_synthetic_bars(csv_path, periods=80)
    _write_model(model_path, p_edge=0.1, orientation="direct")

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    rc = main(
        [
            "--csv",
            str(csv_path),
            "--simulation-mode",
            "--allow-shorts",
            "--model-path",
            str(model_path),
            "--confidence-threshold",
            "0.0",
            "--entry-score-threshold",
            "0.05",
            "--output-json",
            str(out_path),
        ]
    )

    assert rc == 0
    payload = _load_json(out_path)
    assert payload["aggregate"]["model_score"]["supports_short_scores"] is False
    assert int(payload["aggregate"]["total_trades"]) == 0
    assert not any(
        intent["side"] in {"sell", "sell_short"}
        for intent in payload["replay"]["intents"]
    )


def test_offline_replay_model_scoring_applies_negative_symbol_penalties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "PEN.csv"
    base_model = tmp_path / "base_model.joblib"
    penalized_model = tmp_path / "penalized_model.joblib"
    out_base = tmp_path / "base.json"
    out_penalized = tmp_path / "penalized.json"
    _write_synthetic_bars(csv_path, periods=120)
    _write_model(base_model, p_edge=0.9, orientation="direct")
    _write_model(
        penalized_model,
        p_edge=0.9,
        orientation="direct",
        penalties={
            "PEN": {
                "threshold_bump": 0.40,
                "confidence_scale": 0.40,
                "negative_share": 0.25,
            }
        },
    )

    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    args = [
        "--csv",
        str(csv_path),
        "--simulation-mode",
        "--no-allow-shorts",
        "--confidence-threshold",
        "0.0",
        "--entry-score-threshold",
        "0.0",
    ]
    assert main(args + ["--model-path", str(base_model), "--output-json", str(out_base)]) == 0
    assert (
        main(args + ["--model-path", str(penalized_model), "--output-json", str(out_penalized)])
        == 0
    )

    base_payload = _load_json(out_base)
    penalized_payload = _load_json(out_penalized)
    assert int(base_payload["aggregate"]["total_trades"]) > int(
        penalized_payload["aggregate"]["total_trades"]
    )
    assert penalized_payload["aggregate"]["model_score"]["symbol_penalty_count"] == 1


def test_offline_replay_applies_regime_threshold_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "REG.csv"
    model_path = tmp_path / "model.joblib"
    thresholds_path = tmp_path / "regime_thresholds.json"
    out_path = tmp_path / "regime.json"
    accepted_dir = tmp_path / "accepted"
    _write_synthetic_bars(csv_path, periods=120)
    _write_model(model_path, p_edge=0.9, orientation="direct")
    thresholds_path.write_text(
        json.dumps(
            {
                "artifact_type": "regime_thresholds",
                "generated_at": "2026-05-01T15:00:00Z",
                "status": {"available": True, "status": "ready"},
                "regimes": [
                    {
                        "regime": "opening",
                        "confidence_threshold": 0.95,
                        "entry_score_threshold": 0.0,
                        "sample_count": 50,
                    },
                    {
                        "regime": "midday",
                        "confidence_threshold": 0.0,
                        "entry_score_threshold": 0.0,
                        "sample_count": 50,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_PROBABILITY", "1.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", "0")

    assert (
        main(
            [
                "--csv",
                str(csv_path),
                "--simulation-mode",
                "--no-allow-shorts",
                "--model-path",
                str(model_path),
                "--regime-thresholds-json",
                str(thresholds_path),
                "--confidence-threshold",
                "0.0",
                "--entry-score-threshold",
                "0.0",
                "--export-accepted-candidates",
                "--accepted-candidates-dir",
                str(accepted_dir),
                "--output-json",
                str(out_path),
            ]
        )
        == 0
    )

    payload = _load_json(out_path)
    assert payload["aggregate"]["config"]["regime_thresholds"]["enabled"] is True
    rows = [
        json.loads(line)
        for line in (accepted_dir / "accepted_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert {row["session_regime"] for row in rows} == {"midday"}
    assert {row["threshold_source"] for row in rows} == {"regime_threshold:midday"}
    by_regime = payload["aggregate"]["candidate_quality"]["by_session_regime"]
    assert by_regime[0]["session_regime"] == "midday"
