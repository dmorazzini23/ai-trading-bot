"""Cloud/infra MCP server for host checks and controlled restart actions."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional at import time
    psutil = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))
utc_now_iso = cast(Callable[[], str], getattr(_mcp_common_mod, "utc_now_iso"))

_DEFAULT_RUNTIME_ROOT = Path("/var/lib/ai-trading-bot/runtime")
_DEFAULT_AUDIT_PATH = _DEFAULT_RUNTIME_ROOT / "infra_actions_audit.jsonl"


def _bool_arg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _audit_path(args: dict[str, Any]) -> Path:
    raw = (
        str(args.get("audit_path") or "").strip()
        or os.getenv("AI_TRADING_INFRA_AUDIT_PATH", "").strip()
    )
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_AUDIT_PATH


def _append_audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _run_cmd(cmd: list[str], timeout_s: int = 45) -> dict[str, Any]:
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


def _http_get_text(
    *,
    url: str,
    timeout_s: float = 1.5,
    headers: dict[str, str] | None = None,
) -> str:
    request = urllib.request.Request(url=url, method="GET")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = cast(bytes, response.read())
        return body.decode("utf-8", errors="replace").strip()


def _aws_metadata(path: str, timeout_s: float) -> str | None:
    token_request = urllib.request.Request(
        url="http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
    )
    token = None
    try:
        with urllib.request.urlopen(token_request, timeout=timeout_s) as response:
            token = response.read().decode("utf-8", errors="replace")
    except Exception:
        token = None
    headers = {"X-aws-ec2-metadata-token": token} if token else None
    try:
        return _http_get_text(
            url=f"http://169.254.169.254/latest/meta-data/{path}",
            timeout_s=timeout_s,
            headers=headers,
        )
    except Exception:
        return None


def tool_host_summary(_: dict[str, Any]) -> dict[str, Any]:
    disk = shutil.disk_usage("/")
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "disk_total_bytes": int(disk.total),
        "disk_used_bytes": int(disk.used),
        "disk_free_bytes": int(disk.free),
    }
    if psutil is not None:
        vm = psutil.virtual_memory()
        payload.update(
            {
                "memory_total_bytes": int(vm.total),
                "memory_available_bytes": int(vm.available),
                "memory_percent": float(vm.percent),
                "boot_time_unix": float(psutil.boot_time()),
                "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
            }
        )
    return payload


def tool_service_status(args: dict[str, Any]) -> dict[str, Any]:
    unit = str(args.get("unit") or "ai-trading")
    result = _run_cmd(
        ["systemctl", "show", unit, "-p", "ActiveState,SubState,MainPID,ExecMainStatus"],
        timeout_s=20,
    )
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "systemctl show failed")
    fields: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return {"unit": unit, "state": fields}


def tool_journal_errors(args: dict[str, Any]) -> dict[str, Any]:
    unit = str(args.get("unit") or "ai-trading")
    since = str(args.get("since") or "45 min ago")
    limit = max(1, int(args.get("limit") or 120))
    result = _run_cmd(["journalctl", "-u", unit, "--since", since, "-o", "cat"], timeout_s=30)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "journalctl failed")
    rows = [
        line
        for line in result["stdout"].splitlines()
        if any(token in line.lower() for token in ("error", "critical", "traceback"))
    ]
    return {"unit": unit, "since": since, "count": len(rows), "lines": rows[-limit:]}


def tool_controlled_restart(args: dict[str, Any]) -> dict[str, Any]:
    unit = str(args.get("unit") or "ai-trading")
    reason = str(args.get("reason") or "").strip()
    actor = str(args.get("actor") or "mcp_infra_cloud_server").strip()
    confirm = _bool_arg(args.get("confirm"), default=False)
    if not confirm:
        return {
            "executed": False,
            "reason": "set {'confirm': true} to execute restart",
            "unit": unit,
        }

    result = _run_cmd(["systemctl", "restart", unit], timeout_s=60)
    success = result["rc"] == 0
    audit = {
        "ts": utc_now_iso(),
        "action": "service_restart",
        "unit": unit,
        "actor": actor,
        "reason": reason or None,
        "success": success,
        "stderr": result["stderr"].strip() or None,
    }
    _append_audit(_audit_path(args), audit)
    if not success:
        raise RuntimeError(result["stderr"].strip() or "systemctl restart failed")
    return {"executed": True, "unit": unit, "audit_path": str(_audit_path(args))}


def tool_restart_audit_tail(args: dict[str, Any]) -> dict[str, Any]:
    path = _audit_path(args)
    limit = max(1, int(args.get("limit") or 25))
    if not path.exists():
        return {"audit_path": str(path), "count": 0, "rows": []}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return {"audit_path": str(path), "count": len(rows), "rows": rows[-limit:]}


def tool_metadata_probe(args: dict[str, Any]) -> dict[str, Any]:
    provider = str(args.get("provider") or "auto").strip().lower()
    timeout_s = float(args.get("timeout_s") or 1.5)
    candidates = [provider] if provider != "auto" else ["digitalocean", "aws", "gcp"]

    for candidate in candidates:
        try:
            if candidate == "digitalocean":
                data = {
                    "id": _http_get_text(url="http://169.254.169.254/metadata/v1/id", timeout_s=timeout_s),
                    "hostname": _http_get_text(
                        url="http://169.254.169.254/metadata/v1/hostname",
                        timeout_s=timeout_s,
                    ),
                    "region": _http_get_text(
                        url="http://169.254.169.254/metadata/v1/region",
                        timeout_s=timeout_s,
                    ),
                }
                return {"provider": candidate, "detected": True, "metadata": data}
            if candidate == "aws":
                instance_id = _aws_metadata("instance-id", timeout_s)
                if instance_id:
                    data = {
                        "instance_id": instance_id,
                        "instance_type": _aws_metadata("instance-type", timeout_s),
                        "availability_zone": _aws_metadata("placement/availability-zone", timeout_s),
                    }
                    return {"provider": candidate, "detected": True, "metadata": data}
            if candidate == "gcp":
                headers = {"Metadata-Flavor": "Google"}
                project_id = _http_get_text(
                    url="http://169.254.169.254/computeMetadata/v1/project/project-id",
                    timeout_s=timeout_s,
                    headers=headers,
                )
                data = {
                    "project_id": project_id,
                    "zone": _http_get_text(
                        url="http://169.254.169.254/computeMetadata/v1/instance/zone",
                        timeout_s=timeout_s,
                        headers=headers,
                    ),
                }
                return {"provider": candidate, "detected": True, "metadata": data}
        except Exception:
            continue

    return {"provider": provider, "detected": False, "metadata": {}}


TOOLS = {
    "host_summary": tool_host_summary,
    "service_status": tool_service_status,
    "journal_errors": tool_journal_errors,
    "controlled_restart": tool_controlled_restart,
    "restart_audit_tail": tool_restart_audit_tail,
    "metadata_probe": tool_metadata_probe,
}

TOOL_SPECS: list[ToolSpec] = [
    {"name": "host_summary", "description": "Get host CPU/memory/disk summary for infra checks."},
    {"name": "service_status", "description": "Get systemd service state fields via systemctl show."},
    {"name": "journal_errors", "description": "Tail and filter recent service errors from journald."},
    {
        "name": "controlled_restart",
        "description": "Restart service with explicit confirm flag and append audit record.",
    },
    {"name": "restart_audit_tail", "description": "Read recent restart audit records from JSONL file."},
    {"name": "metadata_probe", "description": "Probe cloud metadata endpoint (DO/AWS/GCP)."},
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Cloud/infra MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_infra_cloud",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
