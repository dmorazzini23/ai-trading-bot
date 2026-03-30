#!/usr/bin/env python3
"""Migrate local .env secret values into AWS Secrets Manager."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

try:
    from scripts.runtime_env_sync import _entries_to_map
    from scripts.runtime_env_sync import _infer_secret_key_names
    from scripts.runtime_env_sync import _load_env_entries
    from scripts.runtime_env_sync import _parse_csv_keys
    from scripts.runtime_env_sync import _parse_secret_string
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from runtime_env_sync import _entries_to_map
    from runtime_env_sync import _infer_secret_key_names
    from runtime_env_sync import _load_env_entries
    from runtime_env_sync import _parse_csv_keys
    from runtime_env_sync import _parse_secret_string


def _upsert_env_line(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    replaced = False
    output: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            output.append(f"{key}={value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key}={value}")
    return output


def _aws_cli_base(*, region: str, profile: str) -> list[str]:
    command: list[str] = ["aws"]
    if profile:
        command.extend(["--profile", profile])
    if region:
        command.extend(["--region", region])
    return command


def _run_json(command: list[str]) -> dict[str, Any]:
    aws_env = _aws_cli_env()
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=aws_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "aws CLI not found. Install AWS CLI v2 and ensure `aws` is on PATH."
        ) from exc
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "aws command failed"
        raise RuntimeError(f"{' '.join(command)} failed: {message}")
    try:
        parsed = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command returned invalid JSON: {' '.join(command)}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"command returned non-object JSON: {' '.join(command)}")
    return parsed


def _read_existing_secret_map(secret_id: str, *, region: str, profile: str) -> dict[str, str]:
    command = _aws_cli_base(region=region, profile=profile) + [
        "secretsmanager",
        "get-secret-value",
        "--secret-id",
        secret_id,
        "--output",
        "json",
    ]
    aws_env = _aws_cli_env()
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=aws_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "aws CLI not found. Install AWS CLI v2 and ensure `aws` is on PATH."
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").lower()
        if (
            "resource not found" in stderr
            or "resourcenotfoundexception" in stderr
            or "not found" in stderr
            or "can't find the specified secret" in stderr
        ):
            return {}
        message = proc.stderr.strip() or proc.stdout.strip() or "aws get-secret-value failed"
        raise RuntimeError(f"{' '.join(command)} failed: {message}")
    try:
        parsed = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command returned invalid JSON: {' '.join(command)}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"command returned non-object JSON: {' '.join(command)}")
    secret_string = str(parsed.get("SecretString") or "").strip()
    if not secret_string:
        return {}
    return cast(dict[str, str], _parse_secret_string(secret_string))


def _secret_exists(secret_id: str, *, region: str, profile: str) -> bool:
    command = _aws_cli_base(region=region, profile=profile) + [
        "secretsmanager",
        "describe-secret",
        "--secret-id",
        secret_id,
    ]
    aws_env = _aws_cli_env()
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=aws_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "aws CLI not found. Install AWS CLI v2 and ensure `aws` is on PATH."
        ) from exc
    if proc.returncode == 0:
        return True
    stderr = (proc.stderr or "").lower()
    stdout = (proc.stdout or "").lower()
    combined = f"{stderr}\n{stdout}"
    if (
        "resource not found" in combined
        or "resourcenotfoundexception" in combined
        or "can't find the specified secret" in combined
        or "not found" in combined
    ):
        return False
    message = proc.stderr.strip() or proc.stdout.strip() or "aws describe-secret failed"
    raise RuntimeError(f"{' '.join(command)} failed: {message}")


def _write_secret(secret_id: str, payload: dict[str, str], *, region: str, profile: str) -> None:
    secret_string = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    base = _aws_cli_base(region=region, profile=profile) + [
        "secretsmanager",
    ]
    if _secret_exists(secret_id, region=region, profile=profile):
        command = base + [
            "put-secret-value",
            "--secret-id",
            secret_id,
            "--secret-string",
            secret_string,
        ]
    else:
        command = base + [
            "create-secret",
            "--name",
            secret_id,
            "--secret-string",
            secret_string,
        ]
    _run_json(command)


def _aws_cli_env() -> dict[str, str]:
    env = dict(os.environ)
    candidate_homes: list[Path] = []
    home = str(env.get("HOME") or "").strip()
    if home:
        candidate_homes.append(Path(home))
    try:
        passwd_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        passwd_home = None
    if passwd_home is not None and passwd_home not in candidate_homes:
        candidate_homes.append(passwd_home)
    if not home and candidate_homes:
        env["HOME"] = str(candidate_homes[0])

    creds_already = str(env.get("AWS_SHARED_CREDENTIALS_FILE") or "").strip()
    config_already = str(env.get("AWS_CONFIG_FILE") or "").strip()
    for base in candidate_homes:
        aws_dir = base / ".aws"
        creds_file = aws_dir / "credentials"
        config_file = aws_dir / "config"
        if not creds_already and creds_file.exists():
            env["AWS_SHARED_CREDENTIALS_FILE"] = str(creds_file)
            creds_already = env["AWS_SHARED_CREDENTIALS_FILE"]
        if not config_already and config_file.exists():
            env["AWS_CONFIG_FILE"] = str(config_file)
            config_already = env["AWS_CONFIG_FILE"]
        if creds_already and config_already:
            break
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate .env secret keys to AWS Secrets Manager.")
    parser.add_argument("--env-file", default=".env", help="Path to source env file")
    parser.add_argument("--secret-id", required=True, help="AWS Secrets Manager secret id/name")
    parser.add_argument("--region", default="", help="AWS region override")
    parser.add_argument("--profile", default="", help="AWS profile override")
    parser.add_argument(
        "--managed-keys",
        default="",
        help="Comma-separated explicit secret keys to migrate",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge with existing secret payload instead of replacing it",
    )
    parser.add_argument(
        "--strip-local",
        action="store_true",
        help="Blank managed keys in .env after successful upload",
    )
    parser.add_argument(
        "--write-backend-config",
        action="store_true",
        help="Write AWS backend settings into .env",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute payload and print summary without calling AWS or modifying files.",
    )
    args = parser.parse_args(argv)

    env_path = Path(args.env_file).expanduser().resolve()
    entries = _load_env_entries(env_path)
    env_map = _entries_to_map(entries)
    explicit_keys = _parse_csv_keys(args.managed_keys or env_map.get("AI_TRADING_MANAGED_SECRET_KEYS", ""))
    managed_keys = set(explicit_keys)
    managed_keys.update(_infer_secret_key_names(set(env_map.keys())))

    local_payload = {
        key: value
        for key, value in env_map.items()
        if key in managed_keys and str(value).strip()
    }
    if not local_payload:
        raise RuntimeError("no non-empty managed secret keys found in env file")

    payload = dict(local_payload)
    if args.merge_existing and not args.dry_run:
        existing = _read_existing_secret_map(
            args.secret_id,
            region=args.region,
            profile=args.profile,
        )
        existing.update(payload)
        payload = existing

    if not args.dry_run:
        _write_secret(
            args.secret_id,
            payload,
            region=args.region,
            profile=args.profile,
        )

    backup_path = None
    if (args.strip_local or args.write_backend_config) and not args.dry_run:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        backup_path = env_path.with_name(f"{env_path.name}.bak.{stamp}")
        shutil.copy2(env_path, backup_path)
        lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if args.strip_local:
            for key in sorted(managed_keys):
                if key not in env_map:
                    continue
                lines = _upsert_env_line(lines, key, "")
        if args.write_backend_config:
            lines = _upsert_env_line(lines, "AI_TRADING_SECRETS_BACKEND", "aws-secrets-manager")
            lines = _upsert_env_line(lines, "AI_TRADING_AWS_SECRET_ID", args.secret_id)
            if args.region:
                lines = _upsert_env_line(lines, "AI_TRADING_AWS_REGION", args.region)
            if args.profile:
                lines = _upsert_env_line(lines, "AI_TRADING_AWS_PROFILE", args.profile)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "ok": True,
        "env_file": str(env_path),
        "secret_id": args.secret_id,
        "managed_key_count": len(managed_keys),
        "uploaded_key_count": len(payload),
        "strip_local": bool(args.strip_local),
        "write_backend_config": bool(args.write_backend_config),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "dry_run": bool(args.dry_run),
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            "uploaded",
            summary["uploaded_key_count"],
            "keys to",
            summary["secret_id"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
