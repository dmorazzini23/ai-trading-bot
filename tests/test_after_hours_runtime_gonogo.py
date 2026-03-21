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
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_LOOKBACK_DAYS", "5")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_MIN_USED_DAYS", "3")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_TRADE_FILL_SOURCE", "live")

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
    assert thresholds.get("lookback_days") == 5
    assert thresholds.get("min_used_days") == 3
    assert thresholds.get("trade_fill_source") == "live"
    assert thresholds.get("auto_live_fail_closed") is True


def test_runtime_performance_go_no_go_gate_prefers_after_hours_threshold_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_ENABLED", "1")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_MIN_PROFIT_FACTOR",
        "1.02",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_PROFIT_FACTOR",
        "0.70",
    )

    captured: dict[str, object] = {}

    def _build_report(**_kwargs):
        return {"trade_history": {"pnl_available": True}, "gate_effectiveness": {"valid": True}}

    def _evaluate_go_no_go(report, *, thresholds=None):
        captured["report"] = report
        captured["thresholds"] = dict(thresholds or {})
        return {
            "gate_passed": True,
            "checks": {},
            "failed_checks": [],
            "thresholds": dict(thresholds or {}),
            "observed": {},
        }

    monkeypatch.setattr(rpt, "build_report", _build_report)
    monkeypatch.setattr(rpt, "evaluate_go_no_go", _evaluate_go_no_go)

    result = after_hours._runtime_performance_go_no_go_gate()

    assert result["enabled"] is True
    thresholds = captured["thresholds"]
    assert isinstance(thresholds, dict)
    assert thresholds.get("min_profit_factor") == 1.02


def test_runtime_performance_go_no_go_gate_resolves_runtime_paths(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path / "data-root"))
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH", "runtime/tca_records.jsonl")
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
        "runtime/gate_effectiveness_summary.json",
    )

    captured: dict[str, object] = {}

    def _build_report(*, trade_history_path, gate_summary_path, gate_log_path=None):
        captured["trade_history_path"] = trade_history_path
        captured["gate_summary_path"] = gate_summary_path
        captured["gate_log_path"] = gate_log_path
        return {"trade_history": {"pnl_available": True}, "gate_effectiveness": {"valid": True}}

    monkeypatch.setattr(rpt, "build_report", _build_report)
    monkeypatch.setattr(
        rpt,
        "evaluate_go_no_go",
        lambda *_args, **_kwargs: {
            "gate_passed": True,
            "checks": {},
            "failed_checks": [],
            "thresholds": {},
            "observed": {},
        },
    )

    result = after_hours._runtime_performance_go_no_go_gate()

    assert result["enabled"] is True
    assert captured["trade_history_path"] == (
        tmp_path / "data-root" / "runtime" / "tca_records.jsonl"
    ).resolve()
    assert captured["gate_summary_path"] == (
        tmp_path / "data-root" / "runtime" / "gate_effectiveness_summary.json"
    ).resolve()
    assert captured["gate_log_path"] is None


def test_runtime_performance_go_no_go_gate_defaults_enable_reconciliation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_ENABLED", "1")
    captured: dict[str, object] = {}

    def _build_report(**_kwargs):
        return {"trade_history": {"pnl_available": True}, "gate_effectiveness": {"valid": True}}

    def _evaluate_go_no_go(report, *, thresholds=None):
        captured["report"] = report
        captured["thresholds"] = dict(thresholds or {})
        return {
            "gate_passed": True,
            "checks": {},
            "failed_checks": [],
            "thresholds": dict(thresholds or {}),
            "observed": {},
        }

    monkeypatch.setattr(rpt, "build_report", _build_report)
    monkeypatch.setattr(rpt, "evaluate_go_no_go", _evaluate_go_no_go)

    result = after_hours._runtime_performance_go_no_go_gate()

    assert result["enabled"] is True
    thresholds = captured["thresholds"]
    assert isinstance(thresholds, dict)
    assert thresholds.get("trade_fill_source") == "auto_live"
    assert thresholds.get("auto_live_fail_closed") is True
    assert thresholds.get("require_open_position_reconciliation") is True
