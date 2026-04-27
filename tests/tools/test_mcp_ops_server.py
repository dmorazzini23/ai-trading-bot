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
