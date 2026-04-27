"""Runtime data MCP-style server (local CLI contract).

This module is intentionally read-only and safe for production diagnostics.
"""

from __future__ import annotations

import json
import importlib
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))

_DEFAULT_RUNTIME_ROOT = Path("/var/lib/ai-trading-bot/runtime")
_MODULE_JSON_TIMEOUT_SECONDS = 60.0


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


def _resolve_runtime_root(args: dict[str, Any]) -> Path:
    raw = str(args.get("runtime_root") or _DEFAULT_RUNTIME_ROOT)
    return Path(raw).expanduser().resolve()


def _safe_runtime_path(root: Path, relative_or_abs: str) -> Path:
    candidate = Path(relative_or_abs).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"path escapes runtime root: {candidate}")
    return candidate


def _json_lines(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    if limit <= 0:
        return rows
    return rows[-limit:]


def _extract_json_objects(payload: str | Iterable[str]) -> list[dict[str, Any]]:
    text = payload if isinstance(payload, str) else "\n".join(payload)
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    length = len(text)
    while idx < length:
        char = text[idx]
        if char not in "{[":
            idx += 1
            continue
        try:
            parsed, next_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        idx = next_idx
    return objects


def _run_module_json(module: str, extra_args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-m", module, *extra_args]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_MODULE_JSON_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"module timed out after {_MODULE_JSON_TIMEOUT_SECONDS:g}s: {module}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"module failed rc={proc.returncode}: {module}; stderr={proc.stderr.strip()}"
        )
    objects = _extract_json_objects(proc.stdout)
    if not objects:
        stdout_tail = proc.stdout.strip().splitlines()[-5:]
        raise RuntimeError(
            "no JSON object emitted by "
            f"{module}; stdout_tail={' | '.join(stdout_tail)}; stderr={proc.stderr.strip()}"
        )
    return objects[-1]


def tool_list_runtime_files(args: dict[str, Any]) -> dict[str, Any]:
    root = _resolve_runtime_root(args)
    pattern = str(args.get("pattern") or "*")
    limit = int(args.get("limit") or 200)
    files: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
        if len(files) >= limit:
            break
    return {"runtime_root": str(root), "count": len(files), "files": files}


def tool_read_json(args: dict[str, Any]) -> dict[str, Any]:
    root = _resolve_runtime_root(args)
    path = _safe_runtime_path(root, str(args["path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "payload": payload}


def tool_tail_jsonl(args: dict[str, Any]) -> dict[str, Any]:
    root = _resolve_runtime_root(args)
    path = _safe_runtime_path(root, str(args["path"]))
    limit = int(args.get("limit") or 20)
    rows = _json_lines(path, limit=max(limit, 1))
    return {"path": str(path), "count": len(rows), "rows": rows}


def tool_trade_history_summary(args: dict[str, Any]) -> dict[str, Any]:
    root = _resolve_runtime_root(args)
    configured = str(args.get("path") or "trade_history.parquet")
    path = _safe_runtime_path(root, configured)
    if not path.exists():
        return {"path": str(path), "exists": False}

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        if not _bool_arg(args.get("allow_trusted_pickle"), default=False):
            raise RuntimeError(
                "pickle trade history requires explicit allow_trusted_pickle=true"
            )
        frame = pd.read_pickle(path)
    else:
        raise RuntimeError(f"unsupported trade history format: {path.suffix}")

    summary: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if "fill_source" in frame:
        summary["fill_source_counts"] = {
            str(k): int(v) for k, v in frame["fill_source"].value_counts(dropna=False).items()
        }
    if "pnl" in frame:
        summary["pnl_sum"] = float(frame["pnl"].sum())
        summary["pnl_mean"] = float(frame["pnl"].mean())
    return summary


def tool_runtime_gonogo_status(args: dict[str, Any]) -> dict[str, Any]:
    payload = _run_module_json("ai_trading.tools.runtime_gonogo_status", ["--json"])
    if "gate_passed" not in payload:
        raise RuntimeError("runtime_gonogo_status JSON did not include gate_passed")
    return payload


def tool_runtime_performance_report(args: dict[str, Any]) -> dict[str, Any]:
    extra = ["--json"]
    if bool(args.get("go_no_go", True)):
        extra.append("--go-no-go")
    payload = _run_module_json("ai_trading.tools.runtime_performance_report", extra)
    if "go_no_go" not in payload:
        raise RuntimeError("runtime_performance_report JSON did not include go_no_go")
    return payload


TOOLS = {
    "list_runtime_files": tool_list_runtime_files,
    "read_json": tool_read_json,
    "tail_jsonl": tool_tail_jsonl,
    "trade_history_summary": tool_trade_history_summary,
    "runtime_gonogo_status": tool_runtime_gonogo_status,
    "runtime_performance_report": tool_runtime_performance_report,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "list_runtime_files",
        "description": "List runtime files under the configured runtime root.",
    },
    {
        "name": "read_json",
        "description": "Read a runtime JSON file by path (bounded to runtime root).",
    },
    {
        "name": "tail_jsonl",
        "description": "Read the last N JSON objects from a runtime JSONL file.",
    },
    {
        "name": "trade_history_summary",
        "description": "Summarize runtime trade history parquet/pickle content.",
    },
    {
        "name": "runtime_gonogo_status",
        "description": "Invoke ai_trading.tools.runtime_gonogo_status and return the final JSON.",
    },
    {
        "name": "runtime_performance_report",
        "description": "Invoke ai_trading.tools.runtime_performance_report and return final JSON.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Runtime data MCP-style server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_runtime_data",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
