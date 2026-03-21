from __future__ import annotations

import logging

import pytest

from ai_trading.core import bot_engine


@pytest.fixture(autouse=True)
def _disable_shadow_snapshot_by_default(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_ENABLED", "1")


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
    assert latest["top_n"] == 2
    assert [entry["symbol"] for entry in latest["ranked"]] == ["AAPL", "GOOG"]


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
