from __future__ import annotations

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

