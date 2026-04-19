from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from ai_trading.core.netting_cycle_summary import finalize_netting_cycle_summary


class _LoggerStub:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, message: str, *, extra: dict[str, Any] | None = None) -> None:
        self.info_calls.append((message, dict(extra or {})))


def test_finalize_netting_cycle_summary_updates_analytics_and_logs() -> None:
    recorder = SimpleNamespace(
        decision_gate_counts=Counter({"OK_TRADE": 2, "KILL_SWITCH": 4, "SPREAD_TOO_WIDE": 1}),
        decision_records_total=5,
        decision_observations=[
            {"symbol": "ALL", "gates": ["KILL_SWITCH"], "accepted": False},
            {"symbol": "AAPL", "gates": ["SPREAD_TOO_WIDE"], "accepted": False},
            {"symbol": "MSFT", "gates": ["OK_TRADE"], "accepted": True},
        ],
    )
    logger = _LoggerStub()
    analytics_calls: dict[str, Any] = {}
    throttled_logs: list[dict[str, Any]] = []
    persisted: list[str] = []

    finalize_netting_cycle_summary(
        runtime=SimpleNamespace(),
        state=SimpleNamespace(),
        now=datetime(2026, 4, 19, 15, 30, tzinfo=UTC),
        loop_start=100.0,
        symbols=["AAPL", "MSFT"],
        targets=[object()],
        proposals_total=3,
        proposals_blocked=1,
        orders_attempted=2,
        orders_submitted=1,
        decision_recorder=recorder,
        uncertainty_cycle_events=[{"symbol": "AAPL"}],
        quarantine_enabled=True,
        update_acceptance_rate_governor_state_func=lambda state, **kwargs: analytics_calls.setdefault(
            "acceptance", kwargs
        ),
        gate_effectiveness_exclude_global_halts_func=lambda: True,
        gate_name_is_halt_noise_func=lambda gate: gate == "KILL_SWITCH",
        update_gate_effectiveness_analytics_func=lambda **kwargs: analytics_calls.setdefault(
            "gate_effectiveness", kwargs
        ),
        update_counterfactual_learning_analytics_func=lambda **kwargs: analytics_calls.setdefault(
            "counterfactual", kwargs
        ),
        update_policy_ablation_analytics_func=lambda **kwargs: analytics_calls.setdefault(
            "policy_ablation", kwargs
        ),
        update_uncertainty_capital_analytics_func=lambda **kwargs: analytics_calls.setdefault(
            "uncertainty", kwargs
        ),
        resolve_runtime_info_log_ttl_seconds_func=lambda name, default: default if name else 0.0,
        should_emit_runtime_info_log_func=lambda runtime, key, ttl_seconds: True,
        log_throttled_event_func=lambda _logger, key, **kwargs: throttled_logs.append(
            {"key": key, **kwargs}
        ),
        persist_quarantine_manager_func=lambda state: persisted.append("persisted"),
        monotonic_time_func=lambda: 101.5,
        logger=logger,
        netting_cycle_slo_log_ttl_default=180.0,
    )

    assert analytics_calls["acceptance"] == {
        "decision_records_total": 5,
        "accepted_decisions": 2,
    }
    assert analytics_calls["gate_effectiveness"]["decision_gate_counts"]["KILL_SWITCH"] == 4
    assert analytics_calls["counterfactual"]["observations"] == recorder.decision_observations
    assert analytics_calls["policy_ablation"]["observations"] == recorder.decision_observations
    assert analytics_calls["uncertainty"]["events"] == [{"symbol": "AAPL"}]
    assert logger.info_calls == [
        (
            "DECISION_REJECT_REASON_SUMMARY",
            {
                "records_total": 5,
                "accepted_records": 2,
                "rejected_records": 3,
                "top_reasons": [{"reason": "SPREAD_TOO_WIDE", "count": 1}],
                "unique_reasons": 1,
            },
        )
    ]
    assert throttled_logs == [
        {
            "key": "NETTING_CYCLE_SLO_1:1:1:1",
            "level": 20,
            "extra": {
                "symbols_requested": 2,
                "targets": 1,
                "proposals_total": 3,
                "proposals_blocked": 1,
                "orders_attempted": 2,
                "orders_submitted": 1,
                "orders_skipped": 1,
                "compute_ms": 1500,
            },
            "message": "NETTING_CYCLE_SLO",
        }
    ]
    assert persisted == ["persisted"]


def test_finalize_netting_cycle_summary_uses_raw_gate_counts_when_global_halts_kept() -> None:
    recorder = SimpleNamespace(
        decision_gate_counts=Counter({"OK_TRADE": 1, "KILL_SWITCH": 2}),
        decision_records_total=3,
        decision_observations=cast(list[dict[str, Any]], []),
    )
    logger = _LoggerStub()

    finalize_netting_cycle_summary(
        runtime=SimpleNamespace(),
        state=SimpleNamespace(),
        now=datetime(2026, 4, 19, 15, 30, tzinfo=UTC),
        loop_start=10.0,
        symbols=[],
        targets=[],
        proposals_total=0,
        proposals_blocked=0,
        orders_attempted=0,
        orders_submitted=0,
        decision_recorder=recorder,
        uncertainty_cycle_events=[],
        quarantine_enabled=False,
        update_acceptance_rate_governor_state_func=lambda *args, **kwargs: None,
        gate_effectiveness_exclude_global_halts_func=lambda: False,
        gate_name_is_halt_noise_func=lambda gate: gate == "KILL_SWITCH",
        update_gate_effectiveness_analytics_func=lambda **kwargs: None,
        update_counterfactual_learning_analytics_func=lambda **kwargs: None,
        update_policy_ablation_analytics_func=lambda **kwargs: None,
        update_uncertainty_capital_analytics_func=lambda **kwargs: None,
        resolve_runtime_info_log_ttl_seconds_func=lambda _name, default: default,
        should_emit_runtime_info_log_func=lambda runtime, key, ttl_seconds: key.startswith(
            "DECISION_REJECT_REASON_SUMMARY:"
        ),
        log_throttled_event_func=lambda *args, **kwargs: None,
        persist_quarantine_manager_func=lambda state: None,
        monotonic_time_func=lambda: 10.0,
        logger=logger,
        netting_cycle_slo_log_ttl_default=180.0,
    )

    assert logger.info_calls == [
        (
            "DECISION_REJECT_REASON_SUMMARY",
            {
                "records_total": 3,
                "accepted_records": 1,
                "rejected_records": 2,
                "top_reasons": [{"reason": "KILL_SWITCH", "count": 2}],
                "unique_reasons": 1,
            },
        )
    ]
