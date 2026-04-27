from __future__ import annotations

import pytest

from tools import mcp_observability_server as obs_srv


def test_service_status_rejects_non_allowlisted_unit() -> None:
    with pytest.raises(RuntimeError, match="allowlisted"):
        obs_srv.tool_service_status({"unit": "ssh.service"})


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

    monkeypatch.setattr(obs_srv.urllib.request, "urlopen", _fake_urlopen)

    result = obs_srv.tool_health_probe({})

    assert result["url"] == "http://127.0.0.1:9001/healthz"
    assert captured["url"] == "http://127.0.0.1:9001/healthz"
