from __future__ import annotations

import json
from datetime import UTC, datetime

from ai_trading.core import bot_engine
from ai_trading.core import netting_candidate_rank


def test_session_bucket_from_ts_uses_market_windows() -> None:
    assert (
        bot_engine._session_bucket_from_ts(datetime(2026, 4, 8, 13, 31, tzinfo=UTC))
        == "opening"
    )
    assert (
        bot_engine._session_bucket_from_ts(datetime(2026, 4, 8, 16, 0, tzinfo=UTC))
        == "midday"
    )
    assert (
        bot_engine._session_bucket_from_ts(datetime(2026, 4, 8, 19, 31, tzinfo=UTC))
        == "closing"
    )
    assert (
        bot_engine._session_bucket_from_ts(datetime(2026, 4, 8, 1, 0, tzinfo=UTC))
        == "offhours"
    )


def test_counterfactual_learning_updates_state_from_accepted_and_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "runtime" / "counterfactual_learning_state.json"
    events_path = tmp_path / "runtime" / "counterfactual_learning_events.jsonl"
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_LEARNING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_STATE_PATH", str(state_path))
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_PROPENSITY_ALPHA", "2")
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_PROPENSITY_BETA", "2")
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_MIN_PROPENSITY", "0.05")

    now = datetime(2026, 4, 8, 19, 45, tzinfo=UTC)
    result = bot_engine._update_counterfactual_learning_analytics(
        observations=[
            {
                "symbol": "AAPL",
                "session_bucket": "opening",
                "accepted": True,
                "expected_net_edge_bps": 12.0,
                "edge_proxy_bps": 3.0,
            },
            {
                "symbol": "AAPL",
                "session_bucket": "opening",
                "accepted": False,
                "expected_net_edge_bps": 10.0,
                "edge_proxy_bps": 10.0,
            },
        ],
        now=now,
    )

    assert result["updated"] is True
    assert result["records"] == 2
    assert result["state_persisted"] is True
    assert result["events_persisted"] is True
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    bucket = payload["buckets"]["AAPL:opening"]
    assert bucket["events"] == 2
    assert bucket["accepted"] == 1
    assert bucket["rejected"] == 1
    assert isinstance(payload["global"]["dr_mean_bps"], (int, float))
    assert float(payload["global"]["missed_dr_sum_bps"]) > 0.0
    assert events_path.exists()


def test_counterfactual_learning_bootstraps_state_when_no_observations(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "runtime" / "counterfactual_learning_state.json"
    events_path = tmp_path / "runtime" / "counterfactual_learning_events.jsonl"
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_LEARNING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_STATE_PATH", str(state_path))
    monkeypatch.setenv("AI_TRADING_COUNTERFACTUAL_EVENTS_PATH", str(events_path))

    now = datetime(2026, 4, 8, 20, 10, tzinfo=UTC)
    result = bot_engine._update_counterfactual_learning_analytics(
        observations=[],
        now=now,
    )

    assert result["enabled"] is True
    assert result["updated"] is False
    assert result["records"] == 0
    assert result["state_ready"] is True
    assert result["events_ready"] is True
    assert state_path.exists()
    assert events_path.exists()


def test_dedupe_gate_root_causes_keeps_first_gate_per_root() -> None:
    deduped = bot_engine._dedupe_gate_root_causes(
        [
            "ENTRY_CONSTRAINED_MIN_NOTIONAL_PRECHECK",
            "ENTRY_CONSTRAINED_SYMBOL_REENTRY_COOLDOWN_PRECHECK",
            "LIQ_PARTICIPATION_BLOCK",
            "LIQ_PARTICIPATION_BLOCK_BYPASSED",
            "OK_TRADE",
        ]
    )
    assert deduped == [
        "ENTRY_CONSTRAINED_MIN_NOTIONAL_PRECHECK",
        "LIQ_PARTICIPATION_BLOCK",
        "OK_TRADE",
    ]


def test_lookup_execution_learning_bucket_entry_and_penalty() -> None:
    state = {
        "global": {},
        "buckets": {},
        "symbol_buckets": {
            "BA:midday:unknown:balanced:buy": {
                "samples": 4,
                "mean_slippage_bps": 26.0,
                "mean_net_edge_bps": -12.0,
                "mean_realization_ratio": 0.40,
                "mean_fill_probability": 0.65,
                "fill_rate": 1.0,
                "mean_adverse_selection_risk_bps": 3.0,
            }
        },
    }

    entry, key = netting_candidate_rank._lookup_execution_learning_bucket_entry(
        state=state,
        symbol="BA",
        session_token="midday",
        regime_token="unknown",
        liquidity_role="balanced",
        side="buy",
        min_samples=2,
        safe_float=bot_engine._safe_float,
    )

    assert key == "BA:midday:unknown:balanced:buy"
    assert entry is not None
    penalty = netting_candidate_rank._compute_execution_learning_rank_penalty(
        entry=entry,
        min_samples=2,
        slippage_floor_bps=4.0,
        slippage_weight=0.30,
        negative_edge_weight=0.15,
        adverse_weight=0.10,
        realization_floor=0.85,
        realization_weight_bps=8.0,
        max_penalty_bps=35.0,
        safe_float=bot_engine._safe_float,
    )

    assert penalty["samples"] == 4
    assert penalty["penalty_bps"] > 0.0
    assert penalty["slippage_penalty_bps"] > 0.0
    assert penalty["negative_edge_penalty_bps"] > 0.0
    assert penalty["realization_penalty_bps"] > 0.0
    assert penalty["mean_fill_probability"] == 0.65


def test_lookup_execution_learning_bucket_entry_matches_live_profile_keys() -> None:
    state = {
        "global": {},
        "buckets": {
            "opening:balanced:sell_short": {
                "samples": 4,
                "mean_slippage_bps": 14.0,
                "fill_rate": 0.25,
            }
        },
        "symbol_buckets": {
            "AAPL:opening:trend:balanced:sell_short": {
                "samples": 4,
                "mean_slippage_bps": 18.0,
                "fill_rate": 0.2,
            }
        },
    }

    entry, key = netting_candidate_rank._lookup_execution_learning_bucket_entry(
        state=state,
        symbol="AAPL",
        session_token="opening",
        regime_token="trend",
        liquidity_role="maker",
        side="sell_short",
        min_samples=2,
        safe_float=bot_engine._safe_float,
    )

    assert key == "AAPL:opening:trend:balanced:sell_short"
    assert entry is not None
    assert entry["fill_rate"] == 0.2


def test_load_recent_rejection_concentration_by_symbol_counts_recent_rows(
    tmp_path,
    monkeypatch,
) -> None:
    decision_path = tmp_path / "runtime" / "decision_records.jsonl"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "bar_ts": "2026-04-17T13:30:00+00:00",
            "symbol": "BA",
            "accepted": False,
            "gates_blocking": [
                "PRE_EXECUTION_ORDER_CHECKS_FAILED",
                "SLIPPAGE_CEILING_BLOCK",
                "PORTFOLIO_LOG_GROWTH",
            ],
        },
        {
            "bar_ts": "2026-04-17T13:40:00+00:00",
            "symbol": "BA",
            "accepted": False,
            "gates_blocking": ["CAPACITY_THROTTLE_SCALE"],
        },
        {
            "bar_ts": "2026-04-17T13:45:00+00:00",
            "symbol": "BA",
            "accepted": True,
            "gates_blocking": ["PRE_EXECUTION_ORDER_CHECKS_FAILED"],
        },
        {
            "bar_ts": "2026-04-17T13:50:00+00:00",
            "symbol": "BA",
            "decision_journal": {"accepted": True},
            "gates": ["OK_TRADE"],
        },
        {
            "bar_ts": "2026-04-17T13:55:00+00:00",
            "symbol": "BA",
            "accepted": False,
            "gates": [
                "EXPECTED_CAPTURE_OPTIMIZER",
                "REJECTION_CONCENTRATION_DEWEIGHT",
                "RANK_DOWNSIDE_OVERLAP_CAP",
            ],
        },
        {
            "symbol": "BA",
            "accepted": False,
            "gates_blocking": ["PRE_EXECUTION_ORDER_CHECKS_FAILED"],
        },
        {
            "bar_ts": "2026-04-16T01:00:00+00:00",
            "symbol": "BA",
            "accepted": False,
            "gates_blocking": ["PRE_EXECUTION_ORDER_CHECKS_FAILED"],
        },
        {
            "decision_journal": {
                "symbol": "MSFT",
                "bar_ts": "2026-04-17T13:58:00+00:00",
                "risk_decision": {
                    "accepted": False,
                    "gates": ["PRE_EXECUTION_ORDER_CHECKS_FAILED"],
                },
            },
        },
        {
            "symbol": "NVDA",
            "decision_journal": {
                "bar_ts": "2026-04-17T13:59:00+00:00",
                "risk_decision": {
                    "accepted": True,
                    "gates": ["OK_TRADE"],
                },
            },
        },
        {
            "bar_ts": "2026-04-17T13:59:30+00:00",
            "symbol": "SPY",
            "accepted": False,
            "gates_blocking": ["RISK_PORTFOLIO_HARD_BLOCK"],
        },
    ]
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_DECISION_LOG_PATH", str(decision_path))

    counts = bot_engine._load_recent_rejection_concentration_by_symbol(
        now=datetime(2026, 4, 17, 14, 0, tzinfo=UTC),
        max_records=100,
        lookback_hours=18.0,
    )

    assert counts["BA"]["total"] == 2
    assert counts["BA"]["pre_execution"] == 1
    assert counts["BA"]["slippage"] == 1
    assert counts["BA"]["capacity"] == 1
    assert counts["BA"]["portfolio"] == 0
    assert counts["MSFT"]["total"] == 1
    assert counts["MSFT"]["pre_execution"] == 1
    assert counts["SPY"]["portfolio"] == 1
    assert "NVDA" not in counts


def test_compute_rejection_concentration_penalty_bps_respects_thresholds() -> None:
    low_penalty = netting_candidate_rank._compute_rejection_concentration_penalty_bps(
        counts={"total": 2, "pre_execution": 1, "slippage": 0, "capacity": 0, "portfolio": 0},
        min_count=3,
        scale_bps=0.45,
        max_penalty_bps=24.0,
        safe_float=bot_engine._safe_float,
    )
    assert low_penalty["penalty_bps"] == 0.0

    high_penalty = netting_candidate_rank._compute_rejection_concentration_penalty_bps(
        counts={"total": 12, "pre_execution": 5, "slippage": 2, "capacity": 3, "portfolio": 6},
        min_count=3,
        scale_bps=0.45,
        max_penalty_bps=24.0,
        safe_float=bot_engine._safe_float,
    )
    assert high_penalty["weighted_count"] > 0.0
    assert 0.0 < high_penalty["penalty_bps"] <= 24.0


def test_policy_ablation_update_and_rollback_disables_negative_slice(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "runtime" / "policy_ablation_state.json"
    events_path = tmp_path / "runtime" / "policy_ablation_events.jsonl"
    rollback_path = tmp_path / "runtime" / "policy_rollback_state.json"
    runtime_toggles_path = tmp_path / "runtime" / "policy_runtime_toggles.json"
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_STATE_PATH", str(state_path))
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("AI_TRADING_POLICY_ROLLBACK_STATE_PATH", str(rollback_path))
    monkeypatch.setenv(
        "AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH",
        str(runtime_toggles_path),
    )
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_MIN_EVENTS", "10")
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_STD_PROXY_BPS", "4.0")
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_NEGATIVE_CONFIDENCE", "0.90")
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_SCHEDULE", "daily_first_run")

    now = datetime(2026, 4, 8, 19, 45, tzinfo=UTC)
    observations = [
        {
            "symbol": "AAPL",
            "accepted": False,
            "edge_proxy_bps": -6.0,
            "gates": ["LIQ_PARTICIPATION_BLOCK"],
            "sleeves": ["day"],
        }
        for _ in range(20)
    ]
    update = bot_engine._update_policy_ablation_analytics(observations=observations, now=now)
    assert update["updated"] is True
    assert state_path.exists()
    state = bot_engine.BotState()
    bot_engine._run_policy_ablation_rollback(
        state,
        now=now,
        market_open_now=True,
    )
    assert rollback_path.exists()
    rollback = json.loads(rollback_path.read_text(encoding="utf-8"))
    assert "GATE:LIQUIDITY_PARTICIPATION" in rollback["disabled_slices"]
    assert "SLEEVE:DAY" in rollback["disabled_slices"]
    assert runtime_toggles_path.exists()
    runtime_toggles = json.loads(runtime_toggles_path.read_text(encoding="utf-8"))
    assert "GATE:LIQUIDITY_PARTICIPATION" in runtime_toggles["disabled_slices"]
    assert runtime_toggles["toggles"]["rankers"]["bandit_enabled"] is True


def test_policy_learning_bootstraps_artifacts_without_observations(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "runtime" / "policy_ablation_state.json"
    events_path = tmp_path / "runtime" / "policy_ablation_events.jsonl"
    rollback_path = tmp_path / "runtime" / "policy_rollback_state.json"
    runtime_toggles_path = tmp_path / "runtime" / "policy_runtime_toggles.json"
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_STATE_PATH", str(state_path))
    monkeypatch.setenv("AI_TRADING_POLICY_ABLATION_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("AI_TRADING_POLICY_ROLLBACK_STATE_PATH", str(rollback_path))
    monkeypatch.setenv(
        "AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH",
        str(runtime_toggles_path),
    )
    rollback_path.parent.mkdir(parents=True, exist_ok=True)
    rollback_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-04-08T20:00:00+00:00",
                "disabled_slices": ["RANKER:BANDIT"],
                "diagnostics": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    now = datetime(2026, 4, 8, 20, 15, tzinfo=UTC)
    result = bot_engine._update_policy_ablation_analytics(observations=[], now=now)

    assert result["enabled"] is True
    assert result["updated"] is False
    assert result["state_ready"] is True
    assert result["events_ready"] is True
    assert result["runtime_toggles_ready"] is True
    assert state_path.exists()
    assert events_path.exists()
    assert runtime_toggles_path.exists()
    runtime_toggles = json.loads(runtime_toggles_path.read_text(encoding="utf-8"))
    assert "RANKER:BANDIT" in runtime_toggles["disabled_slices"]


def test_exit_policy_learning_pressure_context_activates_after_samples(monkeypatch) -> None:
    state = bot_engine.BotState()
    now = datetime(2026, 4, 8, 19, 45, tzinfo=UTC)
    monkeypatch.setenv("AI_TRADING_EXEC_EXIT_POLICY_LEARNER_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_EXIT_POLICY_MIN_SAMPLES", "5")

    for idx in range(8):
        bot_engine._update_exit_policy_learning(
            state,
            symbol="AAPL",
            regime="sideways",
            side="long",
            holding_minutes=90.0 + idx,
            outcome_pct=-0.02,
            now=now,
        )

    pressure = bot_engine._exit_policy_pressure_context(
        state,
        symbol="AAPL",
        regime="sideways",
        side="long",
        now=now,
        position_age_seconds=7_200.0,
        expected_edge_bps=0.5,
    )

    assert pressure["active"] is True
    assert pressure["hazard_band"] in {"mid", "long"}
    assert pressure["pressure_score"] > 0.0


def test_exit_policy_state_persists_and_restores(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "runtime" / "exit_policy_state.json"
    monkeypatch.setenv("AI_TRADING_EXIT_POLICY_STATE_PATH", str(state_path))
    monkeypatch.setenv("AI_TRADING_EXEC_EXIT_POLICY_LEARNER_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXIT_POLICY_STATE_PERSIST_MIN_INTERVAL_SEC", "0")

    now = datetime(2026, 4, 8, 19, 45, tzinfo=UTC)
    state = bot_engine.BotState()
    for idx in range(3):
        bot_engine._update_exit_policy_learning(
            state,
            symbol="AAPL",
            regime="trend",
            side="long",
            holding_minutes=25.0 + idx,
            outcome_pct=-0.01,
            now=now,
        )
    assert state_path.exists()
    loaded_state = bot_engine.BotState()
    restored = bot_engine._restore_exit_policy_state(loaded_state, force=True)
    assert restored is True
    buckets = bot_engine._ensure_exit_policy_state(loaded_state)
    assert any(key.startswith("AAPL|") for key in buckets)


def test_uncertainty_capital_analytics_updates_state_and_autotune(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "runtime" / "uncertainty_capital_state.json"
    events_path = tmp_path / "runtime" / "uncertainty_capital_events.jsonl"
    monkeypatch.setenv("AI_TRADING_UNCERTAINTY_CAPITAL_STATE_PATH", str(state_path))
    monkeypatch.setenv("AI_TRADING_UNCERTAINTY_CAPITAL_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("AI_TRADING_UNCERTAINTY_CAPITAL_AUTO_TUNE_MIN_SAMPLES", "20")
    monkeypatch.setenv("AI_TRADING_UNCERTAINTY_CAPITAL_AUTO_TUNE_ENABLED", "1")

    now = datetime(2026, 4, 8, 19, 45, tzinfo=UTC)
    events = []
    for idx in range(40):
        score = 0.2 + (0.02 * (idx % 10))
        events.append(
            {
                "symbol": "AAPL",
                "score": score,
                "effective_score": score,
                "scale": max(0.1, 1.0 - score),
                "scaled": score > 0.3,
                "blocked": False,
            }
        )
    result = bot_engine._update_uncertainty_capital_analytics(events=events, now=now)
    assert result["updated"] is True
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["total_events"] >= 40
    controls = bot_engine._resolve_uncertainty_capital_auto_controls(
        base_weight=0.6,
        base_min_scale=0.35,
        raw_score=0.85,
        state_payload=payload,
    )
    assert controls["enabled"] is True
    assert controls["auto_tuned"] is True
    assert controls["effective_score"] >= 0.0
