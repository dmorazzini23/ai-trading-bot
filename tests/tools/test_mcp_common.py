from __future__ import annotations

import json

import pytest

from tools import mcp_common


def test_parse_args_json_requires_object() -> None:
    with pytest.raises(ValueError):
        mcp_common.parse_args_json('["x"]')


def test_run_tool_server_list_tools(capsys) -> None:
    rc = mcp_common.run_tool_server(
        argv=["--list-tools"],
        description="demo",
        tools={"ping": lambda _: {"pong": True}},
        specs=[{"name": "ping", "description": "demo"}],
        server_name="demo_server",
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tools"][0]["name"] == "ping"
