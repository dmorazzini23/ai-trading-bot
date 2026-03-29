"""Shared helpers for lightweight MCP-style local tool servers.

These utilities intentionally avoid runtime bot imports so they can run in
degraded environments (for example during incidents).
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, TypedDict

_MCP_JSONRPC_MODULE = "tools.mcp_jsonrpc" if __package__ == "tools" else "mcp_jsonrpc"
_mcp_jsonrpc = importlib.import_module(_MCP_JSONRPC_MODULE)


class ToolSpec(TypedDict):
    """Describes a callable tool exposed by a server entrypoint."""

    name: str
    description: str


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def parse_args_json(raw: str | None) -> dict[str, Any]:
    """Parse CLI JSON args and guarantee a dictionary payload."""
    if raw is None or raw.strip() == "":
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--args JSON must decode to an object")
    return parsed


def make_server_parser(description: str) -> argparse.ArgumentParser:
    """Construct a standard parser shared by all server scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print JSON tool catalog for this server.",
    )
    parser.add_argument(
        "--call",
        type=str,
        default=None,
        help="Tool name to execute.",
    )
    parser.add_argument(
        "--args",
        type=str,
        default=None,
        help="JSON object passed to the selected tool.",
    )
    parser.add_argument(
        "--serve-mcp-stdio",
        action="store_true",
        help="Serve true MCP JSON-RPC over stdio (Content-Length framed).",
    )
    return parser


def run_tool_server(
    *,
    argv: list[str] | None,
    description: str,
    tools: Mapping[str, ToolHandler],
    specs: list[ToolSpec],
    server_name: str,
    server_version: str = "0.1.0",
    input_schemas: Mapping[str, dict[str, Any]] | None = None,
) -> int:
    """Run a lightweight, deterministic CLI server contract.

    The contract is intentionally simple:
    - ``--list-tools`` prints ``{"tools": [...]}``.
    - ``--call NAME --args '{"k":"v"}'`` prints
      ``{"ok": true, "tool": NAME, "result": {...}}``.
    """
    parser = make_server_parser(description)
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps({"tools": specs}, sort_keys=True))
        return 0

    if args.serve_mcp_stdio or not args.call:
        jsonrpc_specs = [
            {"name": spec["name"], "description": spec["description"]}
            for spec in specs
        ]
        _mcp_jsonrpc.serve_mcp_stdio(
            server_name=server_name,
            server_version=server_version,
            tools=tools,
            tool_specs=jsonrpc_specs,
            input_schemas=input_schemas,
        )
        return 0

    handler = tools.get(args.call)
    if handler is None:
        known = sorted(tools)
        print(
            json.dumps(
                {
                    "ok": False,
                    "tool": args.call,
                    "error": f"unknown tool: {args.call}",
                    "known_tools": known,
                    "ts": utc_now_iso(),
                },
                sort_keys=True,
            )
        )
        return 2

    try:
        payload = parse_args_json(args.args)
        result = handler(payload)
        if not isinstance(result, dict):
            raise TypeError("tool handlers must return dict[str, Any]")
    except Exception as exc:  # pragma: no cover - explicit CLI error surface
        print(
            json.dumps(
                {
                    "ok": False,
                    "tool": args.call,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "ts": utc_now_iso(),
                },
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "tool": args.call,
                "result": result,
                "ts": utc_now_iso(),
            },
            sort_keys=True,
        )
    )
    return 0
