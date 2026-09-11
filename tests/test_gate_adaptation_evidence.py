import json

import pytest

from ai_trading.core.netting_execution_context import _runtime_gonogo_suppresses_gate_auto_disable


@pytest.mark.parametrize("payload", [None, "{", "[]", "{}", '{"gate_passed":null}', '{"gate_passed":"true"}', '{"go_no_go":{"gate_passed":false}}'])
def test_unverified_evidence_suppresses(tmp_path, monkeypatch, payload):
    path = tmp_path / "report.json"
    if payload is not None:
        path.write_text(payload)
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_SUPPRESS_ON_RUNTIME_GONOGO_FAIL", "1")
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_RUNTIME_GONOGO_PATH", str(path))
    assert _runtime_gonogo_suppresses_gate_auto_disable()[0] is True


def test_explicit_pass_allows_adaptation(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"go_no_go": {"gate_passed": True}}))
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_SUPPRESS_ON_RUNTIME_GONOGO_FAIL", "1")
    monkeypatch.setenv("AI_TRADING_GATE_AUTO_DISABLE_RUNTIME_GONOGO_PATH", str(path))
    assert _runtime_gonogo_suppresses_gate_auto_disable()[0] is False
