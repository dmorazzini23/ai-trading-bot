"""Host heartbeat never confuses degraded trading readiness with host failure."""

from __future__ import annotations

import io
import json
import subprocess
from urllib.error import HTTPError, URLError

import pytest

from ai_trading.tools import host_heartbeat


class _Reply(io.BytesIO):
    def __init__(self, code: int, payload: object) -> None:
        super().__init__(json.dumps(payload).encode())
        self.status = code

    def __enter__(self) -> "_Reply":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_degraded_canonical_health_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    error = HTTPError(host_heartbeat.HEALTH_URL, 503, "degraded", {}, _Reply(503, {"status": "degraded"}))
    monkeypatch.setattr(host_heartbeat, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(error))
    assert host_heartbeat._health_responding()


@pytest.mark.parametrize("reply", [
    _Reply(200, {"status": "healthy"}),
    _Reply(200, {"status": "ready"}),
])
def test_healthy_canonical_health_is_alive(monkeypatch: pytest.MonkeyPatch, reply: _Reply) -> None:
    monkeypatch.setattr(host_heartbeat, "urlopen", lambda *_a, **_k: reply)
    assert host_heartbeat._health_responding()


def test_unstructured_or_unreachable_health_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_heartbeat, "urlopen", lambda *_a, **_k: _Reply(200, {"ok": True}))
    assert not host_heartbeat._health_responding()
    monkeypatch.setattr(host_heartbeat, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(URLError("offline")))
    assert not host_heartbeat._health_responding()


def test_service_and_health_must_both_respond(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_heartbeat.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 1))
    monkeypatch.setattr(host_heartbeat, "_health_responding", lambda: pytest.fail("health should not run"))
    assert host_heartbeat.collect_liveness() == 0
    monkeypatch.setattr(host_heartbeat.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(host_heartbeat, "_health_responding", lambda: True)
    assert host_heartbeat.collect_liveness() == 1


def test_publish_uses_scoped_metric_without_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(host_heartbeat.subprocess, "run", fake_run)
    host_heartbeat.publish_liveness(0)
    assert calls[0][:4] == ["aws", "cloudwatch", "put-metric-data", "--region"]
    assert calls[0][calls[0].index("--namespace") + 1] == host_heartbeat.NAMESPACE
    datum = json.loads(calls[0][-1])[0]
    assert datum["Value"] == 0
    assert datum["Dimensions"] == [{"Name": "Host", "Value": "ai-trading-primary"}]
