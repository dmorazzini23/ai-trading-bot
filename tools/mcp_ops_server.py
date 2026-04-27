"""System ops MCP-style server with strict allowlisted commands."""

from __future__ import annotations

import importlib
import ipaddress
import json
import re
import subprocess
import urllib.error
import urllib.parse
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
_DEFAULT_HEALTH_PORT = 9001
_MIN_PORT = 1
_MAX_PORT = 65535
_ALLOWED_SYSTEMD_UNITS = frozenset(
    {
        "ai-trading",
        "ai-trading.service",
        "ai-trading-api",
        "ai-trading-api.service",
    }
)


def _systemd_unit(args: dict[str, Any]) -> str:
    unit = str(args.get("unit") or "ai-trading").strip()
    if unit not in _ALLOWED_SYSTEMD_UNITS:
        raise RuntimeError(f"systemd unit is not allowlisted: {unit or '<empty>'}")
    return unit


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


def _bounded_port(value: Any) -> int:
    port = _DEFAULT_HEALTH_PORT if value is None or str(value).strip() == "" else int(value)
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise RuntimeError(f"health port out of range: {port}")
    return port


def _health_url(args: dict[str, Any]) -> str:
    raw_url = str(args.get("url") or "").strip()
    if not raw_url:
        port = _bounded_port(args.get("port"))
        return f"http://127.0.0.1:{port}/healthz"

    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme != "http":
        raise RuntimeError("health_probe url must use http")
    if parsed.username or parsed.password:
        raise RuntimeError("health_probe url must not include credentials")
    if parsed.path != "/healthz" or parsed.params or parsed.query or parsed.fragment:
        raise RuntimeError("health_probe url must target exactly /healthz")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise RuntimeError("health_probe url port out of range") from exc
    if parsed_port is None:
        raise RuntimeError("health_probe url must include a port")
    _bounded_port(parsed_port)
    host = parsed.hostname
    if host is None:
        raise RuntimeError("health_probe url must include a host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise RuntimeError("health_probe url host must be a loopback IP literal") from exc
    if not address.is_loopback:
        raise RuntimeError("health_probe url host must be loopback")
    return raw_url


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
    unit = _systemd_unit(args)
    result = _run_cmd(["systemctl", "status", unit, "--no-pager", "-l"], timeout_s=30)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "systemctl status failed")
    lines = result["stdout"].splitlines()[:120]
    return {"unit": unit, "lines": lines}


def tool_service_restart(args: dict[str, Any]) -> dict[str, Any]:
    """Restart service only when explicitly confirmed."""
    unit = _systemd_unit(args)
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
    timeout_s = float(args.get("timeout_s") or 2.0)
    url = _health_url(args)
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
    unit = _systemd_unit(args)
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
