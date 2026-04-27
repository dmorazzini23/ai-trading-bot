from __future__ import annotations

import pytest

from tools import mcp_ops_server as ops_srv


def test_service_restart_requires_confirm() -> None:
    result = ops_srv.tool_service_restart({"unit": "ai-trading", "confirm": False})
    assert result["executed"] is False
    assert "confirm" in result["reason"]


def test_service_status_parses_output(monkeypatch) -> None:
    def _fake_run_cmd(cmd: list[str], timeout_s: int = 120):
        return {
            "cmd": cmd,
            "rc": 0,
            "stdout": "line1\nline2\nline3\n",
            "stderr": "",
        }

    monkeypatch.setattr(ops_srv, "_run_cmd", _fake_run_cmd)
    result = ops_srv.tool_service_status({"unit": "ai-trading"})
    assert result["unit"] == "ai-trading"
    assert result["lines"] == ["line1", "line2", "line3"]


def test_service_unit_must_be_allowlisted() -> None:
    with pytest.raises(RuntimeError, match="allowlisted"):
        ops_srv.tool_service_status({"unit": "ssh.service"})


def test_health_probe_defaults_to_packaged_shared_port(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def read(self) -> bytes:
            return b'{"ok": true}'

    def _fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(ops_srv.urllib.request, "urlopen", _fake_urlopen)

    result = ops_srv.tool_health_probe({})

    assert result["url"] == "http://127.0.0.1:9001/healthz"
    assert captured["url"] == "http://127.0.0.1:9001/healthz"


@pytest.mark.parametrize(
    "args",
    [
        {"url": "http://169.254.169.254:80/healthz"},
        {"url": "http://127.0.0.1:9001/admin"},
        {"url": "http://127.0.0.1:9001/healthz?target=http://169.254.169.254"},
        {"url": "https://127.0.0.1:9001/healthz"},
        {"url": "http://localhost:9001/healthz"},
        {"port": 0},
        {"port": 65536},
    ],
)
def test_health_probe_rejects_non_loopback_healthz_targets(monkeypatch, args) -> None:
    def _fake_urlopen(*_args, **_kwargs):
        raise AssertionError("urlopen should not be called for rejected health_probe target")

    monkeypatch.setattr(ops_srv.urllib.request, "urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError):
        ops_srv.tool_health_probe(args)


def test_health_probe_allows_explicit_loopback_healthz_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def read(self) -> bytes:
            return b'{"ok": true}'

    def _fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(ops_srv.urllib.request, "urlopen", _fake_urlopen)

    result = ops_srv.tool_health_probe({"url": "http://127.0.0.1:8081/healthz"})

    assert result["status_code"] == 200
    assert captured["url"] == "http://127.0.0.1:8081/healthz"
