from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ops_shift_check.py"
_SPEC = importlib.util.spec_from_file_location("ops_shift_check", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
ops_shift = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops_shift)


def test_build_shift_summary_midday(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHCHECK_PORT", "19001")
    observed_health_args = []

    def record_health_probe(args):
        observed_health_args.append(args)
        return {"ok": True}

    monkeypatch.setattr(
        ops_shift.ops_srv,
        "tool_health_probe",
        record_health_probe,
    )
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
    assert observed_health_args == [{"port": 19001}]


def test_resolve_health_port_defaults_to_packaged_api_port(monkeypatch) -> None:
    monkeypatch.delenv("HEALTHCHECK_PORT", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    assert ops_shift._resolve_health_port() == 9001


def test_resolve_health_port_uses_api_port_when_health_port_absent(monkeypatch) -> None:
    monkeypatch.delenv("HEALTHCHECK_PORT", raising=False)
    monkeypatch.setenv("API_PORT", "19001")

    assert ops_shift._resolve_health_port() == 19001
