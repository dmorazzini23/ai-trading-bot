"""Minimal MCP JSON-RPC 2.0 stdio transport.

This module provides a strict MCP-compatible transport loop so servers can be
consumed by clients that expect native MCP framing and methods.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO, Callable

JsonObject = dict[str, Any]
ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

_SUPPORTED_PROTOCOLS = ("2025-03-26", "2024-11-05", "2024-10-07")


def _choose_protocol_version(client_version: Any) -> str:
    client = str(client_version or "").strip()
    if client in _SUPPORTED_PROTOCOLS:
        return client
    return _SUPPORTED_PROTOCOLS[0]


def _jsonrpc_success(request_id: Any, result: JsonObject) -> JsonObject:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> JsonObject:
    payload: JsonObject = {"code": int(code), "message": message}
    if data is not None:
        payload["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": payload}


def _build_tools_payload(
    *,
    tool_specs: Sequence[Mapping[str, str]],
    input_schemas: Mapping[str, JsonObject] | None,
) -> list[JsonObject]:
    tools: list[JsonObject] = []
    for spec in tool_specs:
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        schema = (
            dict(input_schemas[name])
            if input_schemas and name in input_schemas
            else {"type": "object", "additionalProperties": True}
        )
        tools.append(
            {
                "name": name,
                "description": str(spec.get("description") or "").strip(),
                "inputSchema": schema,
            }
        )
    return tools


def process_mcp_message(
    *,
    message: JsonObject,
    tools: Mapping[str, ToolHandler],
    tool_specs: Sequence[Mapping[str, str]],
    input_schemas: Mapping[str, JsonObject] | None,
    server_name: str,
    server_version: str,
    state: dict[str, Any],
) -> tuple[JsonObject | None, bool]:
    """Process one MCP JSON-RPC message.

    Returns ``(response_or_none, should_exit)``.
    """
    if not isinstance(message, dict):
        return None, False

    method = str(message.get("method") or "").strip()
    request_id = message.get("id")
    is_request = "id" in message
    params = message.get("params")
    params_obj = params if isinstance(params, dict) else {}

    if not method:
        if is_request:
            return _jsonrpc_error(request_id, -32600, "Invalid Request: method is required"), False
        return None, False

    if method == "exit":
        return None, True

    if method == "notifications/initialized":
        return None, False

    if method == "initialized":
        return None, False

    if method == "shutdown":
        state["shutdown"] = True
        if is_request:
            return _jsonrpc_success(request_id, {}), False
        return None, False

    if state.get("shutdown") and method != "exit":
        if is_request:
            return _jsonrpc_error(request_id, -32000, "Server is shut down"), False
        return None, False

    if method == "ping":
        if is_request:
            return _jsonrpc_success(request_id, {}), False
        return None, False

    if method == "initialize":
        init_result: JsonObject = {
            "protocolVersion": _choose_protocol_version(params_obj.get("protocolVersion")),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": server_name, "version": server_version},
        }
        if is_request:
            return _jsonrpc_success(request_id, init_result), False
        return None, False

    if method == "tools/list":
        list_result: JsonObject = {
            "tools": _build_tools_payload(tool_specs=tool_specs, input_schemas=input_schemas)
        }
        if is_request:
            return _jsonrpc_success(request_id, list_result), False
        return None, False

    if method == "tools/call":
        tool_name = str(params_obj.get("name") or "").strip()
        raw_args = params_obj.get("arguments")
        tool_args = raw_args if isinstance(raw_args, dict) else {}

        if not tool_name:
            missing_name_result: JsonObject = {
                "isError": True,
                "content": [{"type": "text", "text": "tools/call requires params.name"}],
            }
            if is_request:
                return _jsonrpc_success(request_id, missing_name_result), False
            return None, False

        handler = tools.get(tool_name)
        if handler is None:
            unknown_tool_result: JsonObject = {
                "isError": True,
                "content": [{"type": "text", "text": f"unknown tool: {tool_name}"}],
            }
            if is_request:
                return _jsonrpc_success(request_id, unknown_tool_result), False
            return None, False

        try:
            tool_result = handler(tool_args)
            if not isinstance(tool_result, dict):
                raise TypeError("tool handler must return dict[str, Any]")
            call_result: JsonObject = {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(tool_result, sort_keys=True)}],
                "structuredContent": tool_result,
            }
        except Exception as exc:  # pragma: no cover - runtime path
            call_result = {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": str(exc), "error_type": type(exc).__name__},
                            sort_keys=True,
                        ),
                    }
                ],
            }
        if is_request:
            return _jsonrpc_success(request_id, call_result), False
        return None, False

    if is_request:
        return _jsonrpc_error(request_id, -32601, f"Method not found: {method}"), False
    return None, False


def _read_headers(stream: BinaryIO) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            if not headers:
                return None
            raise RuntimeError("unexpected EOF while reading MCP headers")
        if line in (b"\r\n", b"\n"):
            return headers
        decoded = line.decode("utf-8", errors="strict").strip()
        if ":" not in decoded:
            raise RuntimeError(f"invalid MCP header line: {decoded!r}")
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()


def _read_framed_message(stream: BinaryIO) -> JsonObject | None:
    headers = _read_headers(stream)
    if headers is None:
        return None
    raw_length = headers.get("content-length")
    if raw_length is None:
        raise RuntimeError("missing Content-Length header")
    length = int(raw_length)
    payload = stream.read(length)
    if len(payload) != length:
        raise RuntimeError("unexpected EOF while reading MCP payload")
    decoded = payload.decode("utf-8", errors="strict")
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise RuntimeError("MCP payload must decode to a JSON object")
    return parsed


def _write_framed_message(stream: BinaryIO, message: JsonObject) -> None:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(payload)
    stream.flush()


def serve_mcp_stdio(
    *,
    server_name: str,
    server_version: str,
    tools: Mapping[str, ToolHandler],
    tool_specs: Sequence[Mapping[str, str]],
    input_schemas: Mapping[str, JsonObject] | None = None,
) -> None:
    """Serve MCP JSON-RPC over stdio until EOF/exit."""
    in_stream = sys.stdin.buffer
    out_stream = sys.stdout.buffer
    state: dict[str, Any] = {"shutdown": False}

    while True:
        request = _read_framed_message(in_stream)
        if request is None:
            break
        response, should_exit = process_mcp_message(
            message=request,
            tools=tools,
            tool_specs=tool_specs,
            input_schemas=input_schemas,
            server_name=server_name,
            server_version=server_version,
            state=state,
        )
        if response is not None:
            _write_framed_message(out_stream, response)
        if should_exit:
            break
