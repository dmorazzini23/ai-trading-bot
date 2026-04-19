from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core.netting_rank_prelude import (
    apply_policy_runtime_overrides,
    load_replay_quality_state,
)


def test_load_replay_quality_state_prefers_fresh_state_payload() -> None:
    now = datetime(2026, 4, 19, 15, 0, tzinfo=UTC)
    state = SimpleNamespace(
        replay_symbol_summary={
            "AAPL": {"sample_count": 10, "net_edge_bps": 12.0, "win_rate": 0.6},
        },
        replay_bucket_summary={
            "by_symbol_session": {
                "AAPL:opening": {"sample_count": 5, "net_edge_bps": 11.0},
            },
            "by_symbol_session_regime": {
                "AAPL:opening:trend": {"sample_count": 3, "net_edge_bps": 10.0},
            },
        },
        replay_symbol_summary_updated_at=now.isoformat(),
    )

    result = load_replay_quality_state(
        state=state,
        now=now,
        enabled=True,
        weight=0.18,
        max_age_hours=24.0,
        auto_disable_if_stale=True,
        get_env=lambda _key, default=None, cast=None: default,
        safe_float=lambda value: float(value) if value is not None else None,
        parse_iso_timestamp=lambda value: datetime.fromisoformat(str(value)),
        resolve_runtime_artifact_path_func=lambda path, **_kwargs: path,
        load_latest_replay_quality_summaries_func=lambda *_args, **_kwargs: ({}, {}, {}, {"source": "file"}),
    )

    assert result.by_symbol["AAPL"]["net_edge_bps"] == 12.0
    assert result.by_symbol_session["AAPL:opening"]["sample_count"] == 5.0
    assert result.by_symbol_session_regime["AAPL:opening:trend"]["net_edge_bps"] == 10.0
    assert result.context["source"] == "state"
    assert result.effective_weight == 0.18


def test_load_replay_quality_state_auto_disables_when_missing() -> None:
    result = load_replay_quality_state(
        state=SimpleNamespace(),
        now=datetime(2026, 4, 19, 15, 0, tzinfo=UTC),
        enabled=True,
        weight=0.25,
        max_age_hours=24.0,
        auto_disable_if_stale=True,
        get_env=lambda _key, default=None, cast=None: default,
        safe_float=lambda value: float(value) if value is not None else None,
        parse_iso_timestamp=lambda value: None,
        resolve_runtime_artifact_path_func=lambda path, **_kwargs: path,
        load_latest_replay_quality_summaries_func=lambda *_args, **_kwargs: ({}, {}, {}, {"source": "file"}),
    )

    assert result.by_symbol == {}
    assert result.effective_weight == 0.0
    assert result.context["auto_disabled"] is True


def test_apply_policy_runtime_overrides_disables_rankers_and_roots() -> None:
    state = apply_policy_runtime_overrides(
        load_policy_runtime_toggles_func=lambda: {
            "disabled_slices": ["RANKER:BANDIT", "GATE:SPREAD_GUARD"],
            "toggles": {
                "rankers": {
                    "counterfactual_enabled": False,
                    "geometric_enabled": False,
                    "portfolio_log_growth_enabled": True,
                },
                "disabled_sleeves": ["swing"],
            },
        },
        bandit_enabled=True,
        counterfactual_enabled=True,
        geometric_tiebreak_enabled=True,
        portfolio_log_growth_rank_enabled=True,
    )

    assert state.bandit_enabled is False
    assert state.counterfactual_enabled is False
    assert state.geometric_tiebreak_enabled is False
    assert state.portfolio_log_growth_rank_enabled is True
    assert "SPREAD_GUARD" in state.disabled_gate_roots
    assert "swing" in state.disabled_sleeves
