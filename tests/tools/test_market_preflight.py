import io
import json
from urllib.error import HTTPError, URLError

import pytest

from ai_trading.tools import market_preflight as preflight


class Response(io.BytesIO):
    code = 200


@pytest.mark.parametrize("status,ok,verdict", [
    (200, True, "ready"), (200, False, "blocked"),
    (503, False, "blocked"), (503, True, "blocked"),
])
def test_valid_readiness_report_does_not_fail_command(monkeypatch, capsys, status, ok, verdict):
    payload = {"ok": ok, "status": "healthy", "reason": "required_model_stale",
               "replay_live_parity_gate": {"ok": False},
               "attention_flags": ["required_model_stale"]}

    def fetch(url, timeout):
        assert url == preflight.HEALTH_URL and timeout == 15
        body = json.dumps(payload).encode()
        if status == 503:
            raise HTTPError(url, status, "Unavailable", {}, io.BytesIO(body))
        return Response(body)

    monkeypatch.setattr(preflight, "urlopen", fetch)
    assert preflight.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["readiness_verdict"] == verdict
    assert result["http_status"] == status
    assert result["replay_live_parity_gate"] == {"ok": False}
    assert result["attention_flags"] == payload["attention_flags"]


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b'{}', b'{"ok":"false"}'])
def test_invalid_health_is_unknown_not_ready(monkeypatch, capsys, body):
    monkeypatch.setattr(preflight, "urlopen", lambda *a, **k: Response(body))
    assert preflight.main() == 1
    assert json.loads(capsys.readouterr().out)["readiness_verdict"] == "unknown"


def test_transport_failure_reports_cause(monkeypatch, capsys):
    def fetch(*args, **kwargs):
        raise URLError("connection refused")
    monkeypatch.setattr(preflight, "urlopen", fetch)
    assert preflight.main() == 1
    assert "connection refused" in json.loads(capsys.readouterr().out)["error"]


def test_unexpected_http_status_is_not_a_readiness_report(monkeypatch, capsys):
    def fetch(url, **kwargs):
        raise HTTPError(url, 500, "Error", {}, io.BytesIO(b'{"ok":false}'))
    monkeypatch.setattr(preflight, "urlopen", fetch)
    assert preflight.main() == 1
    assert "500" in json.loads(capsys.readouterr().out)["error"]
