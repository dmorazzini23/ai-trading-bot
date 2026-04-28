from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, cast

from ai_trading.core.netting_execution_context import build_netting_execution_context
from ai_trading.policy.compiler import SafetyTier


class _LoggerStub:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, dict[str, object] | None]] = []

    def info(self, event: str, *, extra: dict[str, object] | None = None) -> None:
        self.entries.append(("info", event, extra))

    def warning(self, event: str, *, extra: dict[str, object] | None = None) -> None:
        self.entries.append(("warning", event, extra))

    def debug(self, event: str, *args: object, **kwargs: object) -> None:
        _ = args, kwargs
        self.entries.append(("debug", event, None))

    def log(self, level: int, event: str, *, extra: dict[str, object] | None = None) -> None:
        _ = level
        self.entries.append(("log", event, extra))


def _base_kwargs() -> dict[str, Any]:
    return {
        "cfg": SimpleNamespace(execution_mode="live"),
        "state": SimpleNamespace(
            operational_safety_tier=SafetyTier.NORMAL.value,
            capital_ramp_multiplier=0.8,
            burn_in_ready=False,
            burn_in_block_reason="BURN_IN",
            _last_capacity_throttle_adaptive_signature=None,
            halt_trading=False,
            halt_reason="",
        ),
        "runtime": SimpleNamespace(),
        "now": datetime(2026, 4, 19, 14, 30, tzinfo=UTC),
        "targets": {"AAPL": SimpleNamespace(), "MSFT": SimpleNamespace()},
        "positions": {"AAPL": 10.0, "MSFT": -5.0},
        "latest_price": {"AAPL": 100.0, "MSFT": 200.0},
        "blocked_symbols": {"AAPL"},
        "candidate_expected_net_edge": {"AAPL": 12.0, "MSFT": 6.0},
        "allocation_weights": {"alpha": 1.0},
        "learned_overrides": {},
        "sleeve_snapshot": {"alpha": {"enabled": True}},
        "effective_policy": object(),
        "kill_switch": False,
        "logger": _LoggerStub(),
        "policy_disabled_gate_roots": {"POLICY_GATE"},
        "decision_record_config_snapshot_func": lambda **kwargs: {"snapshot": True, **kwargs},
        "execution_model_lineage_func": lambda: {"model": "unit"},
        "pretrade_rate_limiter_func": lambda state: {"limiter": id(state)},
        "tca_stale_block_reason_func": lambda now: None,
        "resolve_slo_derisk_effective_mode_func": lambda **kwargs: ("scale", 0.7, {"derived": True}),
        "resolve_operational_safety_tier_func": lambda policy, metrics, previous: (
            SafetyTier.SAFE,
            ("SAFE_TEST",),
        ),
        "apply_operational_safety_hysteresis_func": lambda **kwargs: (
            kwargs["candidate_tier"],
            kwargs["candidate_reasons"],
        ),
        "update_rollout_governance_state_func": lambda **kwargs: {"capital_ramp": {"phase_index": 2}},
        "resolve_capacity_throttle_adaptive_params_func": lambda **kwargs: (
            12.0,
            30.0,
            0.05,
            0.20,
            0.25,
            {"enabled": True, "mode": "steady"},
        ),
        "resolve_primary_feed_derisk_state_func": lambda runtime: {"triggered": False},
        "resolve_runtime_info_log_ttl_seconds_func": lambda name, default: default,
        "should_emit_runtime_info_log_func": lambda *args, **kwargs: False,
        "read_jsonl_records_func": lambda path, max_records: [],
        "gate_effectiveness_log_path_func": lambda: Path("runtime/gate_effectiveness.jsonl"),
        "apply_gate_auto_disable_hysteresis_func": lambda **kwargs: (
            kwargs["candidate_disabled_gates"],
            kwargs["candidate_diagnostics"],
            {
                "min_on_dwell_sec": 0.0,
                "min_off_dwell_sec": 0.0,
                "min_disabled_hold_sec": 0.0,
                "max_transitions_per_hour": 0,
                "transitions_used_in_window": 0,
                "transitions": [],
                "holds": [],
                "candidate_count": 0,
                "effective_count": 0,
            },
        ),
        "symbol_adaptive_sizing_profiles_func": lambda state, symbols: {
            str(symbol): {"scale": 0.9} for symbol in symbols
        },
        "get_sector_func": lambda symbol: "tech" if symbol == "AAPL" else "finance",
        "load_uncertainty_capital_state_func": lambda: {"version": 1},
    }


def test_build_netting_execution_context_collects_global_controls(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", "0")
    kwargs = _base_kwargs()
    context = build_netting_execution_context(**cast(Any, kwargs))

    assert context.decision_snapshot_template["snapshot"] is True
    assert context.execution_model_lineage == {"model": "unit"}
    assert context.ramp_summary == {"phase_index": 2}
    assert context.live_execution_mode is True
    assert context.portfolio_current_gross == 2000.0
    assert context.sector_gross == {"TECH": 1000.0, "FINANCE": 1000.0}
    assert context.symbol_adaptive_profiles["AAPL"]["scale"] == 0.9
    assert "POLICY_GATE" in context.ineffective_gate_blocklist
    assert context.ineffective_gate_diagnostics["POLICY_GATE"]["blocked_records"] == 0.0
    assert context.uncertainty_capital_state == {"version": 1}


def test_build_netting_execution_context_keeps_participation_gates_critical(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_NON_POSITIVE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_LOOKBACK_CYCLES", "10")
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_MIN_BLOCKED", "10")
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_MIN_CONTRIBUTION_BPS", "0")
    kwargs = _base_kwargs()
    kwargs["policy_disabled_gate_roots"] = {
        "LIQUIDITY_PARTICIPATION",
        "LIQ_PARTICIPATION_BLOCK",
        "CAPACITY_THROTTLE_BLOCK",
        "POLICY_GATE",
    }
    kwargs["read_jsonl_records_func"] = lambda path, max_records: [
        {
            "gate_attribution": {
                "LIQ_PARTICIPATION_BLOCK": {
                    "blocked_records": 100,
                    "edge_proxy_bps_sum": 50.0,
                },
                "CAPACITY_THROTTLE_BLOCK": {
                    "blocked_records": 100,
                    "edge_proxy_bps_sum": 50.0,
                },
                "NON_CRITICAL_GATE": {
                    "blocked_records": 100,
                    "edge_proxy_bps_sum": 50.0,
                },
            }
        }
    ]

    context = build_netting_execution_context(**cast(Any, kwargs))

    assert "NON_CRITICAL_GATE" in context.ineffective_gate_blocklist
    assert "POLICY_GATE" in context.ineffective_gate_blocklist
    assert "LIQ_PARTICIPATION_BLOCK" not in context.ineffective_gate_blocklist
    assert "LIQUIDITY_PARTICIPATION" not in context.ineffective_gate_blocklist
    assert "CAPACITY_THROTTLE_BLOCK" not in context.ineffective_gate_blocklist


def test_build_netting_execution_context_blocks_on_slo_breach(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DERISK_SLO_MIN_SAMPLES", "1")
    monitor = SimpleNamespace(
        get_slo_status=lambda key: {
            "order_reject_rate_pct": {"current_value": 10.0, "sample_count": 2},
        }.get(key, {"current_value": 0.0, "sample_count": 0})
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_trading.monitoring.slo",
        SimpleNamespace(get_slo_monitor=lambda: monitor),
    )
    kwargs = _base_kwargs()
    state = kwargs["state"]
    kwargs["resolve_slo_derisk_effective_mode_func"] = lambda **kwargs: (
        "block",
        1.0,
        {"reason": "reject_rate"},
    )

    context = build_netting_execution_context(**cast(Any, kwargs))

    assert getattr(state, "halt_trading") is True
    assert getattr(state, "halt_reason") == "DERISK_SLO_BREACH_BLOCK"
    assert context.slo_derisk_details["breached"] is True
    assert context.slo_derisk_details["effective_mode"] == "block"
