"""Secrets manager MCP server for AWS-backed runtime secret workflows."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_sync_mod = importlib.import_module("scripts.runtime_env_sync")
_load_env_entries = cast(
    Callable[[Path], list[Any]],
    getattr(_sync_mod, "_load_env_entries"),
)
_entries_to_map = cast(
    Callable[[list[Any]], dict[str, str]],
    getattr(_sync_mod, "_entries_to_map"),
)
_parse_csv_keys = cast(
    Callable[[str | None], set[str]],
    getattr(_sync_mod, "_parse_csv_keys"),
)
_infer_secret_key_names = cast(
    Callable[[set[str]], set[str]],
    getattr(_sync_mod, "_infer_secret_key_names"),
)
_parse_secret_string = cast(
    Callable[[str], dict[str, str]],
    getattr(_sync_mod, "_parse_secret_string"),
)
_aws_cli_env = cast(Callable[[], dict[str, str]], getattr(_sync_mod, "_aws_cli_env"))

_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"
_DEFAULT_RUNTIME_ENV_FILE = _REPO_ROOT / ".env.runtime"


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


def _env_file(path_arg: Any, default: Path) -> Path:
    raw = str(path_arg or "").strip()
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


def _env_map(path: Path) -> dict[str, str]:
    entries = _load_env_entries(path)
    return _entries_to_map(entries)


def _run_cmd(cmd: list[str], timeout_s: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=timeout_s,
        env=_aws_cli_env(),
    )
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _aws_secret_id(args: dict[str, Any], env_map: dict[str, str]) -> str:
    secret_id = (
        str(args.get("secret_id") or "").strip()
        or str(env_map.get("AI_TRADING_AWS_SECRET_ID") or "").strip()
    )
    if secret_id:
        return secret_id
    raise RuntimeError("missing secret_id (arg or AI_TRADING_AWS_SECRET_ID)")


def _aws_region(args: dict[str, Any], env_map: dict[str, str]) -> str:
    return (
        str(args.get("region") or "").strip()
        or str(env_map.get("AI_TRADING_AWS_REGION") or "").strip()
        or str(os.getenv("AWS_REGION") or "").strip()
        or "us-west-2"
    )


def _aws_profile(args: dict[str, Any], env_map: dict[str, str]) -> str:
    return (
        str(args.get("profile") or "").strip()
        or str(env_map.get("AI_TRADING_AWS_PROFILE") or "").strip()
        or str(os.getenv("AWS_PROFILE") or "").strip()
    )


def _aws_cmd_base(*, region: str, profile: str) -> list[str]:
    cmd = ["aws"]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    return cmd


def tool_secrets_backend_status(args: dict[str, Any]) -> dict[str, Any]:
    env_file = _env_file(args.get("env_file"), _DEFAULT_ENV_FILE)
    runtime_env_file = _env_file(args.get("runtime_env_file"), _DEFAULT_RUNTIME_ENV_FILE)
    env_map = _env_map(env_file)
    runtime_map = _env_map(runtime_env_file)

    backend = str(env_map.get("AI_TRADING_SECRETS_BACKEND") or "none").strip().lower()
    secret_id = str(env_map.get("AI_TRADING_AWS_SECRET_ID") or "").strip()
    explicit_managed = _parse_csv_keys(env_map.get("AI_TRADING_MANAGED_SECRET_KEYS"))
    inferred_managed = _infer_secret_key_names(set(env_map.keys()))
    managed_keys = sorted(explicit_managed | inferred_managed)
    runtime_managed_present = [key for key in managed_keys if runtime_map.get(key)]

    return {
        "env_file": str(env_file),
        "runtime_env_file": str(runtime_env_file),
        "backend": backend or "none",
        "secret_id": secret_id or None,
        "configured": backend not in {"", "none", "off", "disabled"} and bool(secret_id),
        "managed_key_count": len(managed_keys),
        "managed_keys": managed_keys,
        "runtime_managed_present_count": len(runtime_managed_present),
    }


def tool_aws_identity(args: dict[str, Any]) -> dict[str, Any]:
    env_file = _env_file(args.get("env_file"), _DEFAULT_ENV_FILE)
    env_map = _env_map(env_file)
    region = _aws_region(args, env_map)
    profile = _aws_profile(args, env_map)
    cmd = _aws_cmd_base(region=region, profile=profile) + [
        "sts",
        "get-caller-identity",
        "--output",
        "json",
    ]
    result = _run_cmd(cmd, timeout_s=20)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "aws sts get-caller-identity failed")
    payload = json.loads(result["stdout"] or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("aws sts output was not a JSON object")
    return {
        "region": region,
        "profile": profile or None,
        "identity": payload,
    }


def tool_aws_secret_inventory(args: dict[str, Any]) -> dict[str, Any]:
    env_file = _env_file(args.get("env_file"), _DEFAULT_ENV_FILE)
    env_map = _env_map(env_file)
    secret_id = _aws_secret_id(args, env_map)
    region = _aws_region(args, env_map)
    profile = _aws_profile(args, env_map)

    cmd = _aws_cmd_base(region=region, profile=profile) + [
        "secretsmanager",
        "get-secret-value",
        "--secret-id",
        secret_id,
        "--output",
        "json",
    ]
    result = _run_cmd(cmd, timeout_s=30)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "aws secretsmanager get-secret-value failed")
    payload = json.loads(result["stdout"] or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("aws get-secret-value output was not a JSON object")

    secret_string = str(payload.get("SecretString") or "").strip()
    secret_map = _parse_secret_string(secret_string) if secret_string else {}
    key_lengths = {key: len(value) for key, value in secret_map.items()}

    explicit_managed = _parse_csv_keys(env_map.get("AI_TRADING_MANAGED_SECRET_KEYS"))
    inferred_managed = _infer_secret_key_names(set(env_map.keys()))
    managed_keys = sorted(explicit_managed | inferred_managed)
    missing_managed = [key for key in managed_keys if key not in secret_map]

    return {
        "secret_id": secret_id,
        "region": region,
        "profile": profile or None,
        "arn": payload.get("ARN"),
        "version_id": payload.get("VersionId"),
        "key_count": len(secret_map),
        "keys": sorted(secret_map.keys()),
        "key_lengths": key_lengths,
        "managed_key_count": len(managed_keys),
        "managed_missing_count": len(missing_managed),
        "managed_missing_keys": missing_managed,
    }


def tool_sync_runtime_env(args: dict[str, Any]) -> dict[str, Any]:
    cmd = [str(_REPO_ROOT / "scripts" / "sync_env_runtime.sh")]
    result = _run_cmd(cmd, timeout_s=60)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "sync_env_runtime failed")
    return result


def tool_migrate_local_env_to_aws(args: dict[str, Any]) -> dict[str, Any]:
    if not _bool_arg(args.get("confirm"), default=False):
        return {
            "executed": False,
            "reason": "set {'confirm': true} to execute migration",
        }

    env_file = _env_file(args.get("env_file"), _DEFAULT_ENV_FILE)
    env_map = _env_map(env_file)
    secret_id = _aws_secret_id(args, env_map)
    region = _aws_region(args, env_map)
    profile = _aws_profile(args, env_map)

    cmd = [
        "python3",
        str(_REPO_ROOT / "scripts" / "migrate_secrets_to_aws_sm.py"),
        "--env-file",
        str(env_file),
        "--secret-id",
        secret_id,
        "--region",
        region,
        "--json",
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if _bool_arg(args.get("merge_existing"), default=True):
        cmd.append("--merge-existing")
    if _bool_arg(args.get("strip_local"), default=False):
        cmd.append("--strip-local")
    if _bool_arg(args.get("write_backend_config"), default=True):
        cmd.append("--write-backend-config")
    if _bool_arg(args.get("dry_run"), default=False):
        cmd.append("--dry-run")

    result = _run_cmd(cmd, timeout_s=180)
    if result["rc"] != 0:
        raise RuntimeError(result["stderr"].strip() or "migrate_secrets_to_aws_sm failed")

    payload = json.loads(result["stdout"] or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("migration output was not a JSON object")
    return {"executed": True, "summary": payload}


TOOLS = {
    "secrets_backend_status": tool_secrets_backend_status,
    "aws_identity": tool_aws_identity,
    "aws_secret_inventory": tool_aws_secret_inventory,
    "sync_runtime_env": tool_sync_runtime_env,
    "migrate_local_env_to_aws": tool_migrate_local_env_to_aws,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "secrets_backend_status",
        "description": "Inspect runtime secrets-backend configuration and managed key coverage.",
    },
    {
        "name": "aws_identity",
        "description": "Run aws sts get-caller-identity using current backend profile/region.",
    },
    {
        "name": "aws_secret_inventory",
        "description": "Read AWS secret metadata + key names/lengths (no secret values).",
    },
    {
        "name": "sync_runtime_env",
        "description": "Render .env.runtime using scripts/sync_env_runtime.sh.",
    },
    {
        "name": "migrate_local_env_to_aws",
        "description": "Run migrate_secrets_to_aws_sm.py (requires {'confirm': true}).",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Secrets manager MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_secrets_manager",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
