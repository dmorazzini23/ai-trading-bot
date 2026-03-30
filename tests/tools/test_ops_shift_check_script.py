from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ops_shift_check.py"
_SPEC = importlib.util.spec_from_file_location("ops_shift_check", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
ops_shift = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops_shift)


def test_build_shift_summary_midday(monkeypatch) -> None:
    monkeypatch.setattr(ops_shift.ops_srv, "tool_health_probe", lambda args: {"ok": True})
    monkeypatch.setattr(ops_shift.obs_srv, "tool_service_status", lambda args: {"state": {"ActiveState": "active"}})
    monkeypatch.setattr(ops_shift.obs_srv, "tool_runtime_kpi_snapshot", lambda args: {"gate_passed": True})
    monkeypatch.setattr(
        ops_shift.metrics_srv,
        "tool_execution_trends_snapshot",
        lambda args: {"backend": "prometheus"},
    )
    monkeypatch.setattr(
        ops_shift.market_srv,
        "tool_market_risk_window",
        lambda args: {"risk_level": "normal"},
    )

    payload = ops_shift.build_shift_summary("midday")
    assert payload["phase"] == "midday"
    assert payload["ok"] is True
    assert len(payload["checks"]) >= 5
