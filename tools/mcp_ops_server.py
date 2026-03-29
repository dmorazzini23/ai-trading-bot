"""System ops MCP-style server with strict allowlisted commands."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: list[str], timeout_s: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=timeout_s,
    )
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def tool_sync_env_runtime(_: dict[str, Any]) -> dict[str, Any]:
    result = _run_cmd([str(_REPO_ROOT / "scripts/sync_env_runtime.sh")], timeout_s=45)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "sync_env_runtime failed")
    return result


def tool_refresh_runtime_reports(_: dict[str, Any]) -> dict[str, Any]:
    result = _run_cmd([str(_REPO_ROOT / "scripts/refresh_runtime_reports.sh")], timeout_s=150)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "refresh_runtime_reports failed")
    return result


def tool_service_status(args: dict[str, Any]) -> dict[str, Any]:
    unit = str(args.get("unit") or "ai-trading")
    result = _run_cmd(["systemctl", "status", unit, "--no-pager", "-l"], timeout_s=30)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "systemctl status failed")
    lines = result["stdout"].splitlines()[:120]
    return {"unit": unit, "lines": lines}


def tool_service_restart(args: dict[str, Any]) -> dict[str, Any]:
    """Restart service only when explicitly confirmed."""
    unit = str(args.get("unit") or "ai-trading")
    confirm = bool(args.get("confirm", False))
    if not confirm:
        return {
            "executed": False,
            "reason": "set {'confirm': true} to execute restart",
            "unit": unit,
        }
    result = _run_cmd(["systemctl", "restart", unit], timeout_s=60)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "systemctl restart failed")
    return {"executed": True, "unit": unit}


def tool_health_probe(args: dict[str, Any]) -> dict[str, Any]:
    port = int(args.get("port") or 8081)
    timeout_s = float(args.get("timeout_s") or 2.0)
    url = str(args.get("url") or f"http://127.0.0.1:{port}/healthz")
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return {
                "url": url,
                "status_code": int(response.status),
                "payload": json.loads(body),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"url": url, "status_code": int(exc.code), "error": body}
    except urllib.error.URLError as exc:
        return {"url": url, "status_code": None, "error": str(exc.reason)}


def tool_recent_errors(args: dict[str, Any]) -> dict[str, Any]:
    unit = str(args.get("unit") or "ai-trading")
    since = str(args.get("since") or "30 min ago")
    limit = int(args.get("limit") or 80)
    pattern = str(args.get("pattern") or "ERROR|Traceback|CRITICAL")
    result = _run_cmd(
        ["journalctl", "-u", unit, "--since", since, "-o", "cat"],
        timeout_s=45,
    )
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "journalctl failed")
    expr = re.compile(pattern)
    matched = [line for line in result["stdout"].splitlines() if expr.search(line)]
    return {"unit": unit, "since": since, "count": len(matched), "lines": matched[-limit:]}


TOOLS = {
    "sync_env_runtime": tool_sync_env_runtime,
    "refresh_runtime_reports": tool_refresh_runtime_reports,
    "service_status": tool_service_status,
    "service_restart": tool_service_restart,
    "health_probe": tool_health_probe,
    "recent_errors": tool_recent_errors,
}

TOOL_SPECS: list[ToolSpec] = [
    {"name": "sync_env_runtime", "description": "Run scripts/sync_env_runtime.sh."},
    {
        "name": "refresh_runtime_reports",
        "description": "Run scripts/refresh_runtime_reports.sh for latest runtime rollups.",
    },
    {"name": "service_status", "description": "Read systemctl status output for service unit."},
    {
        "name": "service_restart",
        "description": "Restart service only when args include {'confirm': true}.",
    },
    {"name": "health_probe", "description": "Probe health endpoint and return payload."},
    {
        "name": "recent_errors",
        "description": "Scan recent journal logs and return lines matching error regex.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Ops MCP-style server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_ops",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
