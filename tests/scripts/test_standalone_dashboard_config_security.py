from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_config_server_requires_bearer_auth(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_CONFIG_SERVER_TOKEN", "config-token")
    monkeypatch.delenv("AI_TRADING_OPERATOR_TOKEN_MAP", raising=False)
    config_server = _load_script("config_server_under_test", "config_server.py")
    monkeypatch.setattr(config_server, "request", SimpleNamespace(headers={}))

    payload, status = config_server._require_operator_auth()

    assert status == 401
    assert payload == {"error": "operator authentication required"}


def test_config_server_updates_with_valid_bound_operator(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_OPERATOR_TOKEN_MAP", '{"ops@example.com": "ops-token"}')
    config_server = _load_script("config_server_bound_operator_under_test", "config_server.py")
    monkeypatch.setattr(
        config_server,
        "request",
        SimpleNamespace(
            headers={
                "Authorization": "Bearer ops-token",
                "X-AI-Trading-Operator-Id": "ops@example.com",
            },
        ),
    )

    auth_payload, auth_status = config_server._require_operator_auth()
    validated, error = config_server._validated_payload(
        {"volume_spike_threshold": 2.0, "pyramid_levels": {"high": 0.3}}
    )

    assert auth_payload is None
    assert auth_status == 200
    assert error is None
    assert validated == {
        "volume_spike_threshold": 2.0,
        "ml_confidence_threshold": 0.5,
        "pyramid_levels": {"high": 0.3},
    }


def test_monitoring_dashboard_blocks_remote_without_token(monkeypatch) -> None:
    monkeypatch.delenv("AI_TRADING_DASHBOARD_TOKEN", raising=False)
    monitoring_dashboard = _load_script(
        "monitoring_dashboard_security_under_test",
        "monitoring_dashboard.py",
    )

    assert monitoring_dashboard._remote_addr_is_local("127.0.0.1") is True
    assert monitoring_dashboard._remote_addr_is_local("203.0.113.20") is False


def test_monitoring_dashboard_escapes_alert_and_trade_text() -> None:
    monitoring_dashboard = _load_script(
        "monitoring_dashboard_escape_under_test",
        "monitoring_dashboard.py",
    )
    dashboard = monitoring_dashboard.MonitoringDashboard()

    dashboard.add_alert("warning", "<script>alert(1)</script>", {"x": "<b>bad</b>"})
    dashboard.record_trade("<AAPL>", "<buy>", 1, 10, 2, order_id="<order>")

    alert = list(dashboard.alerts)[-1]
    trade = list(dashboard.trade_history)[-1]
    assert "&lt;script&gt;" in alert.message
    assert trade["symbol"] == "&lt;AAPL&gt;"
    assert trade["order_id"] == "&lt;order&gt;"
