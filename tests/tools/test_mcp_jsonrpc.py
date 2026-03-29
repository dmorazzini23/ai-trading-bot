from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

from tools import mcp_jsonrpc


def _sample_tools() -> dict[str, mcp_jsonrpc.ToolHandler]:
    return {"echo": lambda args: {"echo": args}}


def _sample_specs() -> list[dict[str, str]]:
    return [{"name": "echo", "description": "Echo args back."}]


def test_process_initialize_negotiates_supported_protocol() -> None:
    state: dict[str, Any] = {"shutdown": False}
    response, should_exit = mcp_jsonrpc.process_mcp_message(
        message={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        tools=_sample_tools(),
        tool_specs=_sample_specs(),
        input_schemas=None,
        server_name="demo",
        server_version="1.0.0",
        state=state,
    )
    assert should_exit is False
    assert response is not None
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert response["result"]["serverInfo"]["name"] == "demo"


def test_process_tools_list_uses_input_schemas() -> None:
    state: dict[str, Any] = {"shutdown": False}
    response, _ = mcp_jsonrpc.process_mcp_message(
        message={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        tools=_sample_tools(),
        tool_specs=_sample_specs(),
        input_schemas={"echo": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        server_name="demo",
        server_version="1.0.0",
        state=state,
    )
    assert response is not None
    tools = response["result"]["tools"]
    assert tools[0]["name"] == "echo"
    assert tools[0]["inputSchema"]["properties"]["x"]["type"] == "integer"


def test_process_tools_call_success_contains_structured_content() -> None:
    state: dict[str, Any] = {"shutdown": False}
    response, _ = mcp_jsonrpc.process_mcp_message(
        message={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"k": "v"}},
        },
        tools=_sample_tools(),
        tool_specs=_sample_specs(),
        input_schemas=None,
        server_name="demo",
        server_version="1.0.0",
        state=state,
    )
    assert response is not None
    result = response["result"]
    assert result["isError"] is False
    assert result["structuredContent"] == {"echo": {"k": "v"}}


def test_process_tools_call_unknown_tool_returns_is_error() -> None:
    state: dict[str, Any] = {"shutdown": False}
    response, _ = mcp_jsonrpc.process_mcp_message(
        message={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        },
        tools=_sample_tools(),
        tool_specs=_sample_specs(),
        input_schemas=None,
        server_name="demo",
        server_version="1.0.0",
        state=state,
    )
    assert response is not None
    assert response["result"]["isError"] is True
    assert "unknown tool: missing" in response["result"]["content"][0]["text"]


def test_shutdown_blocks_follow_up_requests() -> None:
    state: dict[str, Any] = {"shutdown": False}
    response, should_exit = mcp_jsonrpc.process_mcp_message(
        message={"jsonrpc": "2.0", "id": 5, "method": "shutdown"},
        tools=_sample_tools(),
        tool_specs=_sample_specs(),
        input_schemas=None,
        server_name="demo",
        server_version="1.0.0",
        state=state,
    )
    assert should_exit is False
    assert response is not None
    assert state["shutdown"] is True

    blocked, _ = mcp_jsonrpc.process_mcp_message(
        message={"jsonrpc": "2.0", "id": 6, "method": "tools/list"},
        tools=_sample_tools(),
        tool_specs=_sample_specs(),
        input_schemas=None,
        server_name="demo",
        server_version="1.0.0",
        state=state,
    )
    assert blocked is not None
    assert blocked["error"]["message"] == "Server is shut down"


def test_framed_round_trip_and_serve_loop(monkeypatch) -> None:
    in_stream = io.BytesIO()
    out_stream = io.BytesIO()
    requests: list[dict[str, Any]] = [
        {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 11, "method": "tools/list"},
        {"jsonrpc": "2.0", "method": "exit"},
    ]
    for request in requests:
        mcp_jsonrpc._write_framed_message(in_stream, request)
    in_stream.seek(0)

    monkeypatch.setattr(mcp_jsonrpc.sys, "stdin", SimpleNamespace(buffer=in_stream))
    monkeypatch.setattr(mcp_jsonrpc.sys, "stdout", SimpleNamespace(buffer=out_stream))

    mcp_jsonrpc.serve_mcp_stdio(
        server_name="demo",
        server_version="1.0.0",
        tools=_sample_tools(),
        tool_specs=_sample_specs(),
    )

    out_stream.seek(0)
    first = mcp_jsonrpc._read_framed_message(out_stream)
    second = mcp_jsonrpc._read_framed_message(out_stream)
    third = mcp_jsonrpc._read_framed_message(out_stream)
    assert first is not None
    assert second is not None
    assert first["id"] == 10
    assert second["id"] == 11
    assert third is None
