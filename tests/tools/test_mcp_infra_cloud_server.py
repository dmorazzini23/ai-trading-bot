from __future__ import annotations

import json
from pathlib import Path

from tools import mcp_infra_cloud_server as infra_srv


def test_controlled_restart_requires_confirm() -> None:
    payload = infra_srv.tool_controlled_restart({"unit": "ai-trading", "confirm": False})
    assert payload["executed"] is False
    assert "confirm" in payload["reason"]


def test_controlled_restart_writes_audit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        infra_srv,
        "_run_cmd",
        lambda cmd, timeout_s=45: {"cmd": cmd, "rc": 0, "stdout": "", "stderr": ""},
    )
    audit_path = tmp_path / "infra_audit.jsonl"
    payload = infra_srv.tool_controlled_restart(
        {
            "confirm": True,
            "unit": "ai-trading",
            "reason": "test_restart",
            "actor": "pytest",
            "audit_path": str(audit_path),
        }
    )
    assert payload["executed"] is True
    rows = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["unit"] == "ai-trading"
    assert parsed["reason"] == "test_restart"


def test_metadata_probe_digitalocean(monkeypatch) -> None:
    responses = {
        "http://169.254.169.254/metadata/v1/id": "12345",
        "http://169.254.169.254/metadata/v1/hostname": "bot-host",
        "http://169.254.169.254/metadata/v1/region": "sfo3",
    }

    def _fake_http_get_text(*, url: str, timeout_s: float = 1.5, headers=None):
        _ = timeout_s, headers
        return responses[url]

    monkeypatch.setattr(infra_srv, "_http_get_text", _fake_http_get_text)
    payload = infra_srv.tool_metadata_probe({"provider": "digitalocean"})
    assert payload["detected"] is True
    assert payload["metadata"]["region"] == "sfo3"
