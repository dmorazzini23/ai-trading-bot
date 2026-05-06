from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.core import bot_engine
from ai_trading.core import netting_candidate_rank

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _disable_shadow_snapshot_by_default(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_ENABLED", "0")


def test_data_retry_settings_clamped_and_logged(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ai_trading.core.bot_engine")
    monkeypatch.setenv("DATA_SOURCE_RETRY_ATTEMPTS", "9")
    monkeypatch.setenv("DATA_SOURCE_RETRY_DELAY_SECONDS", "9.25")
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_LOGGED", False, raising=False)

    attempts, delay = bot_engine._resolve_data_retry_settings()

    assert attempts == 5
    assert delay == 5.0
    matching = [
        record
        for record in caplog.records
        if record.getMessage() == "DATA_RETRY_SETTINGS"
    ]
    assert matching, "expected DATA_RETRY_SETTINGS log"
    assert matching[0].attempts == 5
    assert matching[0].delay_seconds == 5.0


def test_data_retry_settings_flatten_mode(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("DATA_SOURCE_RETRY_DELAY_SECONDS", "2.5")
    monkeypatch.setenv("AI_TRADING_DATA_RETRY_FLATTEN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_RETRY_FLATTEN_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("AI_TRADING_DATA_RETRY_FLATTEN_MAX_DELAY_SECONDS", "0")
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_LOGGED", False, raising=False)

    attempts, delay = bot_engine._resolve_data_retry_settings()

    assert attempts == 1
    assert delay == 0.0


def test_bandit_ucb_score_rewards_uncertainty() -> None:
    low_samples = netting_candidate_rank._bandit_ucb_score(
        mean_reward_bps=2.0,
        samples=4,
        total_samples=200,
        exploration=1.5,
    )
    high_samples = netting_candidate_rank._bandit_ucb_score(
        mean_reward_bps=2.0,
        samples=40,
        total_samples=200,
        exploration=1.5,
    )

    assert low_samples > high_samples
    assert low_samples > 2.0


def test_sequential_significance_gate_requires_min_samples() -> None:
    result = bot_engine._sequential_significance_gate(
        mean_reward_bps=2.0,
        std_reward_bps=5.0,
        samples=9,
        min_samples=10,
        target_mean_bps=0.0,
        method="either",
        posterior_prob_min=0.9,
        sprt_alpha=0.05,
        sprt_beta=0.1,
        sprt_effect_bps=0.4,
    )

    assert result["passed"] is False
    assert result["reason"] == "insufficient_samples"


def test_sequential_significance_gate_passes_with_strong_signal() -> None:
    result = bot_engine._sequential_significance_gate(
        mean_reward_bps=1.8,
        std_reward_bps=2.0,
        samples=200,
        min_samples=20,
        target_mean_bps=0.0,
        method="either",
        posterior_prob_min=0.9,
        sprt_alpha=0.05,
        sprt_beta=0.1,
        sprt_effect_bps=0.4,
    )

    assert result["passed"] is True
    assert result["reason"] == "ok"


def test_geometric_growth_tiebreak_score_penalizes_risk() -> None:
    calm_score = netting_candidate_rank._geometric_growth_tiebreak_score(
        expected_edge_bps=8.0,
        returns_window=[0.002, 0.0015, 0.0018, 0.0012],
        drawdown=0.001,
        variance_penalty=1.0,
        downside_penalty=1.0,
        drawdown_penalty=1.0,
    )
    stressed_score = netting_candidate_rank._geometric_growth_tiebreak_score(
        expected_edge_bps=8.0,
        returns_window=[0.01, -0.02, 0.006, -0.015],
        drawdown=0.03,
        variance_penalty=1.0,
        downside_penalty=1.0,
        drawdown_penalty=1.0,
    )

    assert calm_score > stressed_score


def test_pre_rank_execution_candidates_preserves_input_order_without_weights(monkeypatch):
    monkeypatch.delenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", raising=False)

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=None,
    )

    assert ranked == ["MSFT", "AAPL", "GOOG"]


def test_pre_rank_execution_candidates_dedupes_symbols(monkeypatch):
    monkeypatch.delenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", raising=False)

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "MSFT", "aapl", "GOOG"],
        runtime=None,
    )

    assert ranked == ["MSFT", "AAPL", "GOOG"]


def test_pre_rank_execution_candidates_uses_weights_with_top_n(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "2")
    runtime = type(
        "_Runtime",
        (),
        {"portfolio_weights": {"MSFT": 0.2, "AAPL": 0.4, "GOOG": 0.9}},
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=runtime,
    )

    assert ranked == ["GOOG", "AAPL"]


def test_pre_rank_execution_candidates_prefers_runtime_rank(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "2")
    runtime = type(
        "_Runtime",
        (),
        {
            "portfolio_weights": {"MSFT": 0.8, "AAPL": 0.1, "GOOG": 0.2},
            "execution_candidate_rank": {"MSFT": -5.0, "AAPL": 3.2, "GOOG": 2.1},
        },
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=runtime,
    )

    assert ranked == ["AAPL", "GOOG"]


def test_pre_rank_execution_candidates_can_explore_unseen_symbols(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "2")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_FRAC", "0.5")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_MIN", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_STALE_CYCLES", "1")
    runtime = type(
        "_Runtime",
        (),
        {
            "portfolio_weights": {"MSFT": 0.1, "AAPL": 0.2, "GOOG": 0.3},
            "execution_candidate_rank": {"AAPL": 5.0, "GOOG": 4.0},
        },
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=runtime,
    )

    assert ranked[0] == "AAPL"
    assert "MSFT" in ranked
    assert len(ranked) == 2


def test_pre_rank_execution_candidates_filters_by_opportunity_quality(monkeypatch):
    monkeypatch.delenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", raising=False)
    monkeypatch.setenv("AI_TRADING_EXEC_OPPORTUNITY_QUALITY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE", "0.8")
    monkeypatch.setenv("AI_TRADING_EXEC_OPPORTUNITY_MIN_KEEP", "2")

    runtime = type(
        "_Runtime",
        (),
        {
            "execution_opportunity_quality_by_symbol": {
                "MSFT": 0.96,
                "AAPL": 0.92,
                "GOOG": 0.30,
                "AMZN": 0.20,
                "NVDA": 0.10,
            },
        },
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG", "AMZN", "NVDA"],
        runtime=runtime,
    )

    assert ranked == ["MSFT", "AAPL"]


def test_pre_rank_execution_candidates_records_shadow_snapshot_when_enabled(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "2")
    payloads: list[dict] = []
    monkeypatch.setattr(
        bot_engine,
        "_record_shadow_prediction",
        lambda payload: payloads.append(dict(payload)),
    )
    runtime = type(
        "_Runtime",
        (),
        {
            "portfolio_weights": {"MSFT": 0.8, "AAPL": 0.1, "GOOG": 0.2},
            "execution_candidate_rank": {"MSFT": -5.0, "AAPL": 3.2, "GOOG": 2.1},
        },
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=runtime,
    )

    assert ranked == ["AAPL", "GOOG"]
    assert payloads, "expected prerank shadow snapshot"
    latest = payloads[-1]
    assert latest["mode"] == "execution_candidate_prerank"
    assert latest["rank_source"] == "runtime_rank"
    assert latest["requested"] == 3
    assert latest["selected"] == 2
    assert "provider" in latest
    assert "quote_status" in latest
    assert latest["top_n"] == 2
    assert [entry["symbol"] for entry in latest["ranked"]] == ["AAPL", "GOOG"]


def test_pre_rank_execution_candidates_rotates_single_slot_exploration(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_STALE_CYCLES", "1")
    runtime = type(
        "_Runtime",
        (),
        {
            "execution_candidate_rank": {"AAPL": 10.0, "AMZN": 1.0},
            "_execution_candidate_last_selected_cycle": {"AAPL": 1},
            "_execution_prerank_cycle_idx": 1,
        },
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["AAPL", "AMZN"],
        runtime=runtime,
    )

    assert ranked == ["AMZN"]


def test_pre_rank_execution_candidates_logs_symbol_starvation(monkeypatch, caplog):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_SYMBOL_STARVATION_ALERT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_SYMBOL_STARVATION_WINDOW", "10")
    monkeypatch.setenv("AI_TRADING_SYMBOL_STARVATION_MIN_SAMPLES", "5")
    monkeypatch.setenv("AI_TRADING_SYMBOL_STARVATION_DOMINANCE_RATIO", "0.95")
    monkeypatch.setenv("AI_TRADING_SYMBOL_STARVATION_ALERT_COOLDOWN_CYCLES", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", "AMZN")
    runtime = type(
        "_Runtime",
        (),
        {"execution_candidate_rank": {"AAPL": 10.0}},
    )()
    caplog.set_level(logging.WARNING, logger="ai_trading.core.bot_engine")

    for _ in range(5):
        bot_engine._pre_rank_execution_candidates(["AAPL"], runtime=runtime)

    matching = [
        record
        for record in caplog.records
        if record.getMessage() == "SYMBOL_STARVATION_ALERT"
    ]
    assert matching
    assert matching[-1].dominant_symbol == "AAPL"
    assert "AMZN" in matching[-1].configured_symbols


def test_prerank_ml_signal_shadow_scores_selected_symbols(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_SIGNAL_LIMIT", "2")

    class Model:
        def predict(self, _frame):
            return [1]

        def predict_proba(self, _frame):
            return [[0.2, 0.8]]

    class Fetcher:
        def get_daily_df(self, _runtime, _symbol):
            return pd.DataFrame(
                {
                    "open": [100.0] * 20,
                    "high": [101.0] * 20,
                    "low": [99.0] * 20,
                    "close": [100.0] * 20,
                    "volume": [1000.0] * 20,
                }
            )

    calls: list[tuple[str | None, object]] = []
    manager = bot_engine.SignalManager()
    monkeypatch.setattr(
        manager,
        "signal_ml",
        lambda frame, model=None, symbol=None: calls.append((symbol, model)),
    )
    monkeypatch.setattr(bot_engine, "signal_manager", manager)
    monkeypatch.setattr(bot_engine, "_load_required_model", lambda: Model())
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: None)

    runtime = type("_Runtime", (), {"data_fetcher": Fetcher()})()

    bot_engine._record_prerank_ml_signal_shadow(
        selected_symbols=["AAPL", "MSFT", "GOOG"],
        runtime=runtime,
    )

    assert [symbol for symbol, _model in calls] == ["AAPL", "MSFT"]
    assert all(isinstance(model, Model) for _symbol, model in calls)


def test_prerank_ml_signal_shadow_scores_extra_shadow_symbols(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_SIGNAL_LIMIT", "2")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", "MSFT")

    class Model:
        def predict(self, _frame):
            return [1]

        def predict_proba(self, _frame):
            return [[0.2, 0.8]]

    class Fetcher:
        def get_daily_df(self, _runtime, _symbol):
            return pd.DataFrame(
                {
                    "open": [100.0] * 20,
                    "high": [101.0] * 20,
                    "low": [99.0] * 20,
                    "close": [100.0] * 20,
                    "volume": [1000.0] * 20,
                }
            )

    calls: list[str | None] = []
    manager = bot_engine.SignalManager()
    monkeypatch.setattr(
        manager,
        "signal_ml",
        lambda frame, model=None, symbol=None: calls.append(symbol),
    )
    monkeypatch.setattr(bot_engine, "signal_manager", manager)
    monkeypatch.setattr(bot_engine, "_load_required_model", lambda: Model())
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: None)

    runtime = type("_Runtime", (), {"data_fetcher": Fetcher()})()

    bot_engine._record_prerank_ml_signal_shadow(
        selected_symbols=["AAPL", "AMZN", "GOOG"],
        runtime=runtime,
    )

    assert calls == ["AAPL", "AMZN", "MSFT"]


def test_prerank_ml_signal_shadow_prefetches_symbol_quote(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_SIGNAL_LIMIT", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_QUOTE_PREFETCH_ENABLED", "1")

    class Model:
        def predict(self, _frame):
            return [1]

        def predict_proba(self, _frame):
            return [[0.2, 0.8]]

    class Fetcher:
        def get_daily_df(self, _runtime, _symbol):
            return pd.DataFrame(
                {
                    "open": [100.0] * 20,
                    "high": [101.0] * 20,
                    "low": [99.0] * 20,
                    "close": [100.0] * 20,
                    "volume": [1000.0] * 20,
                }
            )

    class Quote:
        bid_price = 100.0
        ask_price = 100.04
        timestamp = datetime.now(UTC) - timedelta(seconds=2)

    seen_snapshots: list[dict[str, object]] = []
    manager = bot_engine.SignalManager()

    def _record_signal(_frame, model=None, symbol=None):
        seen_snapshots.append(bot_engine.runtime_state.observe_symbol_quote_status(symbol))

    monkeypatch.setattr(manager, "signal_ml", _record_signal)
    monkeypatch.setattr(bot_engine, "signal_manager", manager)
    monkeypatch.setattr(bot_engine, "_load_required_model", lambda: Model())
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: None)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda runtime, symbol: Quote())

    runtime = type("_Runtime", (), {"data_fetcher": Fetcher()})()

    bot_engine.runtime_state.reset_quote_status()
    try:
        bot_engine._record_prerank_ml_signal_shadow(
            selected_symbols=["AAPL"],
            runtime=runtime,
        )
    finally:
        bot_engine.runtime_state.reset_quote_status()

    assert len(seen_snapshots) == 1
    snapshot = seen_snapshots[0]
    assert snapshot["symbol"] == "AAPL"
    assert snapshot["bid"] == pytest.approx(100.0)
    assert snapshot["ask"] == pytest.approx(100.04)
    assert snapshot["quote_age_ms"] == pytest.approx(2000.0, abs=500.0)
    assert snapshot["source"] == "latest_quote"
    assert snapshot["status"] == "ready"


def test_prerank_ml_signal_shadow_prefers_minute_frame(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_SIGNAL_ENABLED", "1")

    class Model:
        def predict(self, _frame):
            return [1]

        def predict_proba(self, _frame):
            return [[0.2, 0.8]]

    class Fetcher:
        def get_daily_df(self, _runtime, _symbol):
            raise AssertionError("daily fallback should not be used when minute frame exists")

    minute_frame = pd.DataFrame(
        {
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000.0] * 20,
        }
    )
    calls: list[object] = []
    manager = bot_engine.SignalManager()
    monkeypatch.setattr(
        manager,
        "signal_ml",
        lambda frame, model=None, symbol=None: calls.append(frame),
    )
    monkeypatch.setattr(bot_engine, "signal_manager", manager)
    monkeypatch.setattr(bot_engine, "_load_required_model", lambda: Model())
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: minute_frame)

    runtime = type("_Runtime", (), {"data_fetcher": Fetcher()})()

    bot_engine._record_prerank_ml_signal_shadow(
        selected_symbols=["AAPL"],
        runtime=runtime,
    )

    assert len(calls) == 1


def test_prepare_prerank_ml_shadow_frame_builds_replay_features() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0 + idx for idx in range(240)],
            "high": [101.0 + idx for idx in range(240)],
            "low": [99.0 + idx for idx in range(240)],
            "close": [100.0 + idx for idx in range(240)],
            "volume": [1000.0] * 240,
        }
    )
    feature_names = [
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
    ]

    prepared = bot_engine._prepare_prerank_ml_shadow_frame(
        frame,
        feature_names=feature_names,
        symbol="AAPL",
    )

    assert set(feature_names).issubset(prepared.columns)
    assert prepared[feature_names].iloc[-1].notna().all()


def test_pre_rank_execution_candidates_adaptive_top_n_from_submit_cap(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "10")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_PER_ORDER_CAP", "3")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_MIN", "2")

    class _ExecEngine:
        @staticmethod
        def _resolve_order_submit_cap():
            return 1, "configured"

    runtime = type("_Runtime", (), {"execution_engine": _ExecEngine()})()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG", "AMZN", "NVDA"],
        runtime=runtime,
    )

    assert ranked == ["MSFT", "AAPL", "GOOG"]


def test_pre_rank_execution_candidates_adaptive_top_n_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "4")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED", "0")

    class _ExecEngine:
        @staticmethod
        def _resolve_order_submit_cap():
            return 1, "configured"

    runtime = type("_Runtime", (), {"execution_engine": _ExecEngine()})()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG", "AMZN", "NVDA"],
        runtime=runtime,
    )

    assert ranked == ["MSFT", "AAPL", "GOOG", "AMZN"]


def test_capacity_throttle_adaptive_params_tighten_on_pacing_stress(monkeypatch):
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_MIN_SAMPLES", "5")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PACING_TIGHTEN_PCT", "20")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_TIGHTEN_MULT", "0.8")

    (
        spread_soft,
        spread_hard,
        vol_soft,
        vol_hard,
        min_scale,
        details,
    ) = bot_engine._resolve_capacity_throttle_adaptive_params(
        spread_soft_bps=12.0,
        spread_hard_bps=30.0,
        volume_soft_participation=0.05,
        volume_hard_participation=0.20,
        min_scale=0.25,
        slo_derisk_details={"pacing_samples": 10, "order_pacing_cap_hit_rate_pct": 35.0},
    )

    assert details["mode"] == "tightened"
    assert spread_soft < 12.0
    assert spread_hard < 30.0
    assert vol_soft < 0.05
    assert vol_hard < 0.20
    assert min_scale <= 0.25


def test_capacity_throttle_adaptive_params_relax_when_quality_is_good(monkeypatch):
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_MIN_SAMPLES", "5")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PACING_CLEAR_PCT", "8")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_RELAX_MULT", "1.2")

    (
        spread_soft,
        spread_hard,
        vol_soft,
        vol_hard,
        min_scale,
        details,
    ) = bot_engine._resolve_capacity_throttle_adaptive_params(
        spread_soft_bps=12.0,
        spread_hard_bps=30.0,
        volume_soft_participation=0.05,
        volume_hard_participation=0.20,
        min_scale=0.25,
        slo_derisk_details={
            "pacing_samples": 10,
            "order_pacing_cap_hit_rate_pct": 1.0,
            "reject_rate_pct": 0.1,
        },
    )

    assert details["mode"] == "relaxed"
    assert spread_soft > 12.0
    assert spread_hard > 30.0
    assert vol_soft > 0.05
    assert vol_hard > 0.20
    assert min_scale >= 0.25


def test_capacity_throttle_adaptive_params_tighten_on_microstructure_stress(monkeypatch):
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_MIN_SAMPLES", "50")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PENDING_SOFT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PENDING_HARD_SEC", "90")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_STRESS_TIGHTEN_MULT", "0.75")
    monkeypatch.setenv("AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_STRESS_MIN_SCALE_MULT", "0.70")

    (
        spread_soft,
        spread_hard,
        vol_soft,
        vol_hard,
        min_scale,
        details,
    ) = bot_engine._resolve_capacity_throttle_adaptive_params(
        spread_soft_bps=12.0,
        spread_hard_bps=30.0,
        volume_soft_participation=0.05,
        volume_hard_participation=0.20,
        min_scale=0.25,
        slo_derisk_details={
            "pacing_samples": 2,
            "order_pacing_cap_hit_rate_pct": 1.0,
            "pending_samples": 3,
            "pending_oldest_age_sec": 120.0,
        },
    )

    assert details["mode"] == "tightened"
    assert details["microstructure_stress"] is True
    assert spread_soft < 12.0
    assert spread_hard < 30.0
    assert vol_soft < 0.05
    assert vol_hard < 0.20
    assert min_scale <= 0.25


def test_slo_derisk_effective_mode_relaxes_block_for_friction_only(monkeypatch):
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_BLOCK_RELAX_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_BLOCK_RELAX_SCALE_MULT", "0.72")
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_BLOCK_RELAX_PACING_SEVERE_PCT", "85")
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_BLOCK_RELAX_PENDING_SEVERE_SEC", "2400")

    mode, scale, details = bot_engine._resolve_slo_derisk_effective_mode(
        configured_mode="block",
        reject_breached=False,
        drift_breached=False,
        slippage_breached=False,
        calibration_ece_breached=False,
        calibration_brier_breached=False,
        feature_drift_breached=False,
        label_drift_breached=False,
        residual_drift_breached=False,
        pacing_breached=True,
        pending_breached=False,
        pacing_hit_rate_pct=35.0,
        pending_oldest_age_sec=0.0,
    )

    assert mode == "scale"
    assert scale == pytest.approx(0.72)
    assert details["block_relax_applied"] is True


def test_slo_derisk_effective_mode_keeps_block_for_core_breach(monkeypatch):
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_BLOCK_RELAX_ENABLED", "1")

    mode, scale, details = bot_engine._resolve_slo_derisk_effective_mode(
        configured_mode="block",
        reject_breached=True,
        drift_breached=False,
        slippage_breached=False,
        calibration_ece_breached=False,
        calibration_brier_breached=False,
        feature_drift_breached=False,
        label_drift_breached=False,
        residual_drift_breached=False,
        pacing_breached=True,
        pending_breached=False,
        pacing_hit_rate_pct=20.0,
        pending_oldest_age_sec=0.0,
    )

    assert mode == "block"
    assert scale == pytest.approx(1.0)
    assert details["block_relax_applied"] is False
    assert details["block_relax_reason"] == "core_breach"


def test_merge_managed_position_symbols_includes_nonzero_positions() -> None:
    merged = bot_engine._merge_managed_position_symbols(
        ["AAPL", "ABBV"],
        {
            "AAPL": 10,
            "MSFT": 5,
            "TSLA": 0,
            "NVDA": float("nan"),
            "AMZN": "3",
        },
    )

    assert merged == ["AAPL", "ABBV", "MSFT", "AMZN"]


def test_load_tickers_falls_back_to_packaged_universe_when_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bot_engine, "load_universe", lambda: ["AAPL", "MSFT"])
    symbols = bot_engine.load_tickers("/tmp/does-not-exist.csv")
    assert symbols == ["AAPL", "MSFT"]
