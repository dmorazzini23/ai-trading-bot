"""Observability MCP-style server (local CLI contract)."""

from __future__ import annotations

import importlib
import json
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

_runtime_data_mod = importlib.import_module(
    "tools.mcp_runtime_data_server"
    if __package__ == "tools"
    else "mcp_runtime_data_server"
)
_run_module_json = cast(
    Callable[[str, list[str]], dict[str, Any]],
    getattr(_runtime_data_mod, "_run_module_json"),
)

_DEFAULT_RUNTIME_ROOT = Path("/var/lib/ai-trading-bot/runtime")


def _run_cmd(cmd: list[str], timeout_s: int = 30) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _runtime_root(args: dict[str, Any]) -> Path:
    raw = str(args.get("runtime_root") or _DEFAULT_RUNTIME_ROOT)
    return Path(raw).expanduser().resolve()


def tool_health_probe(args: dict[str, Any]) -> dict[str, Any]:
    port = int(args.get("port") or 8081)
    timeout_s = float(args.get("timeout_s") or 2.0)
    url = str(args.get("url") or f"http://127.0.0.1:{port}/healthz")
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {
                "url": url,
                "status_code": int(response.status),
                "payload": payload,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"url": url, "status_code": int(exc.code), "error": body}
    except urllib.error.URLError as exc:
        return {"url": url, "status_code": None, "error": str(exc.reason)}


def tool_service_status(args: dict[str, Any]) -> dict[str, Any]:
    unit = str(args.get("unit") or "ai-trading")
    show = _run_cmd(
        ["systemctl", "show", unit, "-p", "ActiveState,SubState,MainPID,ExecMainStatus"]
    )
    if show["rc"] != 0:
        raise RuntimeError(show["stderr"].strip() or "systemctl show failed")
    details: dict[str, Any] = {}
    for line in show["stdout"].splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        details[key] = value
    return {"unit": unit, "state": details}


def tool_journal_tail(args: dict[str, Any]) -> dict[str, Any]:
    unit = str(args.get("unit") or "ai-trading")
    lines = int(args.get("lines") or 200)
    since = str(args.get("since") or "30 min ago")
    cmd = ["journalctl", "-u", unit, "--since", since, "-n", str(lines), "-o", "cat"]
    result = _run_cmd(cmd, timeout_s=45)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "journalctl failed")
    entries = [line for line in result["stdout"].splitlines() if line.strip()]
    return {"unit": unit, "since": since, "line_count": len(entries), "lines": entries}


def tool_runtime_kpi_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    payload = _run_module_json(
        "ai_trading.tools.runtime_performance_report",
        ["--json", "--go-no-go"],
    )
    go_no_go = payload.get("go_no_go") or {}
    execution = payload.get("execution_vs_alpha") or {}
    snapshot = {
        "gate_passed": go_no_go.get("gate_passed"),
        "failed_checks": go_no_go.get("failed_checks"),
        "slippage_drag_bps": execution.get("slippage_drag_bps"),
        "execution_capture_ratio": execution.get("execution_capture_ratio"),
        "execution_drag_share": execution.get("execution_drag_share"),
    }
    return snapshot


def tool_list_runtime_reports(args: dict[str, Any]) -> dict[str, Any]:
    root = _runtime_root(args)
    report_dir = root / "reports"
    out: list[str] = []
    if report_dir.exists():
        for path in sorted(report_dir.glob("*.json")):
            out.append(str(path))
    latest = root / "runtime_performance_report_latest.json"
    if latest.exists():
        out.append(str(latest))
    return {"runtime_root": str(root), "count": len(out), "reports": out}


TOOLS = {
    "health_probe": tool_health_probe,
    "service_status": tool_service_status,
    "journal_tail": tool_journal_tail,
    "runtime_kpi_snapshot": tool_runtime_kpi_snapshot,
    "list_runtime_reports": tool_list_runtime_reports,
}

TOOL_SPECS: list[ToolSpec] = [
    {"name": "health_probe", "description": "Call health endpoint and return JSON payload."},
    {"name": "service_status", "description": "Fetch systemd service state via systemctl show."},
    {"name": "journal_tail", "description": "Tail service logs from journald with structured output."},
    {
        "name": "runtime_kpi_snapshot",
        "description": "Extract go/no-go and execution-vs-alpha KPIs from runtime report tool.",
    },
    {
        "name": "list_runtime_reports",
        "description": "List report JSON files under runtime and runtime/reports.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Observability MCP-style server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_observability",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
