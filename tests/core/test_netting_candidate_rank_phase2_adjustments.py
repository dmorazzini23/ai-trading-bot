from __future__ import annotations

from typing import Any, Sequence

import pytest

from ai_trading.core.netting_candidate_rank import rank_netting_candidates
from tests.test_netting_candidate_rank import _base_kwargs, _make_target


def _mean_std(values: Sequence[float]) -> tuple[float, float, int]:
    clean = [float(value) for value in values]
    if not clean:
        return 0.0, 0.0, 0
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return float(mean), float(variance**0.5), len(clean)


def test_rank_netting_candidates_applies_execution_learning_and_rejection_penalties() -> None:
    kwargs = _base_kwargs()
    kwargs.update(
        {
            "execution_learning_rank_enabled": True,
            "execution_learning_state": {
                "buckets": {
                    "AAPL:opening:trend:maker:buy": {
                        "samples": 8,
                        "mean_slippage_bps": 14.0,
                        "mean_net_edge_bps": -3.0,
                        "mean_adverse_selection_risk_bps": 2.0,
                        "mean_realization_ratio": 0.4,
                        "mean_fill_probability": 0.35,
                    }
                }
            },
            "rejection_concentration_rank_enabled": True,
            "rejection_concentration_by_symbol": {
                "AAPL": {
                    "total": 6,
                    "pre_execution": 2,
                    "slippage": 2,
                    "capacity": 1,
                    "portfolio": 1,
                }
            },
            "expected_capture_learned_fill_model_enabled": True,
            "lookup_learned_fill_probability_func": lambda **_kwargs: (0.8, 12, "learned-fill"),
            "expected_capture_cost_model_enabled": True,
            "lookup_learned_execution_cost_components_func": lambda **_kwargs: (
                1.5,
                2.5,
                0.5,
                20,
                "learned-cost",
            ),
        }
    )

    result = rank_netting_candidates(**kwargs)

    signal = result.counterfactual_signal_by_symbol["AAPL"]
    target = kwargs["targets"]["AAPL"]

    assert "EXPECTED_CAPTURE_MODEL_LEARNED" in target.reasons
    assert "EXECUTION_LEARNING_DEWEIGHT" in target.reasons
    assert "REJECTION_CONCENTRATION_DEWEIGHT" in target.reasons
    assert signal["expected_capture_fill_model_source"] == "learned-fill"
    assert signal["expected_capture_cost_model_source"] == "learned-cost"
    assert signal["execution_learning_source"] == "AAPL:opening:trend:maker:buy"
    assert signal["execution_learning_penalty_bps"] > 0.0
    assert signal["rejection_concentration_penalty_bps"] > 0.0
    assert signal["expected_capture_fill_probability"] == pytest.approx(0.35)


def test_rank_netting_candidates_uses_short_execution_learning_bucket() -> None:
    kwargs = _base_kwargs()
    now = kwargs["now"]
    kwargs.update(
        {
            "targets": {
                "AAPL": _make_target(
                    symbol="AAPL",
                    now=now,
                    target_dollars=-1000.0,
                    confidence=0.9,
                    disagreement=0.8,
                )
            },
            "execution_learning_rank_enabled": True,
            "execution_learning_state": {
                "buckets": {
                    "AAPL:opening:trend:maker:sell_short": {
                        "samples": 8,
                        "mean_slippage_bps": 14.0,
                        "mean_net_edge_bps": -3.0,
                        "mean_adverse_selection_risk_bps": 2.0,
                        "mean_realization_ratio": 0.4,
                        "mean_fill_probability": 0.35,
                    }
                }
            },
        }
    )

    result = rank_netting_candidates(**kwargs)

    signal = result.counterfactual_signal_by_symbol["AAPL"]
    assert signal["execution_learning_source"] == "AAPL:opening:trend:maker:sell_short"
    assert signal["execution_learning_penalty_bps"] > 0.0


def test_rank_netting_candidates_promotes_live_bandit_and_counterfactual_bonus() -> None:
    kwargs = _base_kwargs()
    kwargs.update(
        {
            "mean_std_func": _mean_std,
            "bandit_enabled": True,
            "bandit_live_promoted": True,
            "bandit_shadow_only": False,
            "bandit_bucket_promote_min_samples": 3,
            "bandit_method": "thompson",
            "bandit_rewards_by_symbol_session_regime": {
                "AAPL:opening:trend": [4.0, 5.0, 6.0, 7.0]
            },
            "counterfactual_enabled": True,
            "counterfactual_live_promoted": True,
            "counterfactual_buckets": {
                "AAPL:opening": {"events": 6, "dr_sum_bps": 42.0}
            },
            "sequential_significance_gate_func": lambda **_kwargs: {
                "passed": True,
                "method": "posterior",
            },
        }
    )

    result = rank_netting_candidates(**kwargs)
    signal = result.counterfactual_signal_by_symbol["AAPL"]
    target = kwargs["targets"]["AAPL"]

    assert "BANDIT_THOMPSON_SYMBOL_SESSION_REGIME" in target.reasons
    assert "COUNTERFACTUAL_DR" in target.reasons
    assert signal["bandit_source"] == "symbol_session_regime"
    assert signal["bandit_bucket_live_promoted"] is True
    assert signal["counterfactual_events"] == 6
    assert signal["counterfactual_dr_mean_bps"] == pytest.approx(7.0)
    assert kwargs["bandit_bucket_significance_cache"]


def test_rank_netting_candidates_records_edge_realized_portfolio_replay_and_exit_contexts() -> None:
    kwargs = _base_kwargs()
    target = kwargs["targets"]["AAPL"]
    kwargs.update(
        {
            "mean_std_func": _mean_std,
            "edge_model_v2_enabled": True,
            "realized_edge_rank_enabled": True,
            "portfolio_log_growth_rank_enabled": True,
            "replay_quality_rank_enabled": True,
            "exit_policy_entry_penalty_enabled": True,
            "rank_downside_overlap_cap_enabled": True,
            "rank_downside_overlap_cap_abs": 1.0,
            "edge_model_v2_min_samples": 3,
            "realized_edge_rank_min_samples": 3,
            "realized_edge_by_symbol": {"AAPL": [8.0, 9.0, 10.0]},
            "realized_edge_by_symbol_session": {
                "AAPL:opening": [11.0, 12.0, 13.0]
            },
            "realized_edge_by_symbol_session_regime": {
                "AAPL:opening:trend": [14.0, 15.0, 16.0]
            },
            "portfolio_rank_correlation": {"AAPL": {"MSFT": 0.7}},
            "held_notional_weights": {"MSFT": 0.4},
            "positions": {"AAPL": 2.0},
            "portfolio_current_gross_for_rank": 5000.0,
            "replay_quality_by_symbol_session_regime": {
                "AAPL:opening:trend": {"sample_count": 8, "net_edge_bps": -20.0}
            },
            "replay_quality_max_rank_uplift_abs": 100.0,
            "replay_quality_max_rank_uplift_frac": 10.0,
            "replay_quality_context": {"source": "nightly"},
            "exit_policy_entry_penalty_weight": 50.0,
            "exit_policy_entry_penalty_max_abs": 100.0,
            "exit_policy_pressure_context_func": lambda _state, **_kwargs: {
                "active": True,
                "pressure_score": 0.9,
                "reason": "crowded_exit",
            },
        }
    )

    result = rank_netting_candidates(**kwargs)
    signal = result.counterfactual_signal_by_symbol["AAPL"]

    assert "EDGE_MODEL_V2" in target.reasons
    assert "EDGE_MODEL_V2_REGIME_BLEND" in target.reasons
    assert "PORTFOLIO_LOG_GROWTH" in target.reasons
    assert "REPLAY_QUALITY_DEWEIGHT" in target.reasons
    assert "EXIT_POLICY_ENTRY_PENALTY" in target.reasons
    assert "RANK_DOWNSIDE_OVERLAP_CAP" in target.reasons
    assert signal["edge_model_v2_source"] == "regime+session+global"
    assert signal["realized_rank_source"] == "symbol_session"
    assert signal["portfolio_corr_penalty"] > 0.0
    assert signal["replay_quality_source"] == "nightly:symbol_session_regime"
    assert signal["exit_policy_entry_pressure"] == pytest.approx(0.9)
    assert signal["rank_downside_overlap_cap_applied"] is True
    assert result.candidate_rank["AAPL"] == pytest.approx(
        signal["rank_score_post_adjustments"]
    )


def test_rank_netting_candidates_clips_and_time_decays_negative_edge() -> None:
    kwargs = _base_kwargs()
    now = kwargs["now"]
    kwargs.update(
        {
            "targets": {
                "MSFT": _make_target(
                    symbol="MSFT",
                    now=now,
                    target_dollars=-500.0,
                    confidence=0.8,
                    disagreement=float("nan"),
                )
            },
            "net_edge_raw_by_symbol": {"MSFT": -500.0},
            "latest_liquidity": {},
            "latest_price": {"MSFT": 250.0},
            "symbol_returns": {"MSFT": []},
            "clip_expected_edge_enabled": True,
            "edge_clip_cap_bps": 25.0,
            "alpha_time_decay_enabled": True,
            "alpha_time_decay_half_life_sec": 60.0,
            "alpha_time_decay_floor": 0.25,
            "expected_capture_rank_enabled": False,
        }
    )

    result = rank_netting_candidates(**kwargs)
    signal = result.counterfactual_signal_by_symbol["MSFT"]

    assert result.candidate_expected_net_edge["MSFT"] == -25.0
    assert signal["time_decay_multiplier"] == 1.0
    assert result.candidate_rank["MSFT"] < 0.0
