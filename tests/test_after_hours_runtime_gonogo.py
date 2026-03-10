from __future__ import annotations

from ai_trading.tools import runtime_performance_report as rpt
from ai_trading.training import after_hours


def test_runtime_performance_go_no_go_gate_disabled(monkeypatch) -> None:
    monkeypatch.delenv(
        "AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_ENABLED",
        raising=False,
    )

    result = after_hours._runtime_performance_go_no_go_gate()

    assert result["enabled"] is False
    assert result["gate_passed"] is True
    assert result["reason"] == "disabled"


def test_runtime_performance_go_no_go_gate_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_MIN_CLOSED_TRADES", "25")

    captured: dict[str, object] = {}

    def _build_report(**_kwargs):
        return {"trade_history": {"pnl_available": True}, "gate_effectiveness": {"valid": True}}

    def _evaluate_go_no_go(report, *, thresholds=None):
        captured["report"] = report
        captured["thresholds"] = dict(thresholds or {})
        return {
            "gate_passed": False,
            "checks": {"profit_factor": False},
            "failed_checks": ["profit_factor"],
            "thresholds": dict(thresholds or {}),
            "observed": {"profit_factor": 0.9},
        }

    monkeypatch.setattr(rpt, "build_report", _build_report)
    monkeypatch.setattr(rpt, "evaluate_go_no_go", _evaluate_go_no_go)

    result = after_hours._runtime_performance_go_no_go_gate()

    assert result["enabled"] is True
    assert result["gate_passed"] is False
    assert result["failed_checks"] == ["profit_factor"]
    thresholds = captured["thresholds"]
    assert isinstance(thresholds, dict)
    assert thresholds.get("min_closed_trades") == 25
