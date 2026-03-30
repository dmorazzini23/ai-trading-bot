#!/usr/bin/env python3
"""Render `.env.runtime` from `.env`, with optional secrets-manager overlay."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEY_HINT_RE = re.compile(r"(SECRET|TOKEN|PASSWORD|WEBHOOK_URL$|API_KEY$)")

DEPRECATED_KEYS = {
    "AI_TRADING_EXEC_ALLOW_FALLBACK_WITHOUT_NBBO",
}

_DEFAULT_MANAGED_KEYS = {
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_DATA_KEY",
    "ALPACA_DATA_SECRET_KEY",
    "WEBHOOK_SECRET",
    "AI_TRADING_SLACK_WEBHOOK_URL",
    "SLACK_WEBHOOK_URL",
    "AI_TRADING_LINEAR_API_KEY",
    "LINEAR_API_KEY",
    "AI_TRADING_GRAFANA_API_TOKEN",
    "AI_TRADING_PROM_REMOTE_WRITE_PASSWORD",
    "TRADIER_ACCESS_TOKEN",
    "FINNHUB_API_KEY",
    "SENTIMENT_API_KEY",
    "IEX_API_TOKEN",
    "FRED_API_KEY",
}

_BACKEND_NONE = {"", "none", "off", "disabled"}
_BACKEND_AWS = {"aws-secrets-manager", "aws_sm", "aws"}


@dataclass(frozen=True)
class EnvEntry:
    key: str
    value: str


def _parse_env_entries(text: str) -> list[EnvEntry]:
    entries: list[EnvEntry] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_RE.match(key):
            continue
        entries.append(EnvEntry(key=key, value=value))
    return entries


def _load_env_entries(path: Path) -> list[EnvEntry]:
    if not path.exists():
        return []
    return _parse_env_entries(path.read_text(encoding="utf-8", errors="ignore"))


def _entries_to_map(entries: list[EnvEntry]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        result[entry.key] = entry.value
    return result


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv_keys(raw: str | None) -> set[str]:
    if not raw:
        return set()
    keys: set[str] = set()
    for part in raw.split(","):
        key = part.strip().upper()
        if _ENV_KEY_RE.match(key):
            keys.add(key)
    return keys


def _infer_secret_key_names(keys: set[str]) -> set[str]:
    inferred: set[str] = set()
    for key in keys:
        if key in _DEFAULT_MANAGED_KEYS:
            inferred.add(key)
            continue
        if key == "ALPACA_API_KEY":
            inferred.add(key)
            continue
        if _SECRET_KEY_HINT_RE.search(key):
            inferred.add(key)
    return inferred


def _resolve_setting(
    name: str,
    *,
    env_map: Mapping[str, str],
    default: str = "",
) -> str:
    direct = os.getenv(name)
    if direct is not None and direct.strip():
        return direct.strip()
    return str(env_map.get(name, default) or "").strip()


def _fetch_aws_secret_payload(secret_id: str, *, region: str, profile: str) -> dict[str, str]:
    command: list[str] = ["aws"]
    if profile:
        command.extend(["--profile", profile])
    if region:
        command.extend(["--region", region])
    command.extend(
        [
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            secret_id,
            "--output",
            "json",
        ]
    )
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
        message = proc.stderr.strip() or proc.stdout.strip() or "aws secretsmanager failed"
        raise RuntimeError(f"failed fetching AWS secret '{secret_id}': {message}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AWS secret response was not valid JSON") from exc
    secret_string = str(payload.get("SecretString") or "").strip()
    if not secret_string:
        raise RuntimeError(
            f"AWS secret '{secret_id}' did not include SecretString; SecretBinary is not supported"
        )
    parsed = _parse_secret_string(secret_string)
    if not parsed:
        raise RuntimeError(
            f"AWS secret '{secret_id}' did not contain key/value entries (expected JSON object or dotenv text)"
        )
    return parsed


def _parse_secret_string(raw: str) -> dict[str, str]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {
            str(key).strip().upper(): str(value)
            for key, value in parsed.items()
            if _ENV_KEY_RE.match(str(key).strip().upper()) and value is not None
        }
    entries = _parse_env_entries(text)
    parsed_map = _entries_to_map(entries)
    return {key.upper(): value for key, value in parsed_map.items()}


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


def _render_runtime_env(src: Path, dst: Path) -> dict[str, object]:
    entries = _load_env_entries(src)
    env_map = _entries_to_map(entries)

    backend = _resolve_setting("AI_TRADING_SECRETS_BACKEND", env_map=env_map, default="none").lower()
    secret_id = _resolve_setting("AI_TRADING_AWS_SECRET_ID", env_map=env_map)
    aws_region = _resolve_setting("AI_TRADING_AWS_REGION", env_map=env_map)
    aws_profile = _resolve_setting("AI_TRADING_AWS_PROFILE", env_map=env_map)
    require_managed = _parse_bool(
        _resolve_setting("AI_TRADING_REQUIRE_MANAGED_SECRETS", env_map=env_map, default="0"),
        default=False,
    )

    explicit_managed_keys = _parse_csv_keys(
        _resolve_setting("AI_TRADING_MANAGED_SECRET_KEYS", env_map=env_map)
    )
    managed_keys = set(explicit_managed_keys)
    managed_keys.update(_infer_secret_key_names(set(env_map.keys())))

    manager_values: dict[str, str] = {}
    if backend in _BACKEND_NONE:
        pass
    elif backend in _BACKEND_AWS:
        if not secret_id:
            raise RuntimeError(
                "AI_TRADING_SECRETS_BACKEND is enabled but AI_TRADING_AWS_SECRET_ID is empty"
            )
        manager_values = _fetch_aws_secret_payload(
            secret_id,
            region=aws_region,
            profile=aws_profile,
        )
        managed_keys.update(_infer_secret_key_names(set(manager_values.keys())))
        managed_keys.update(explicit_managed_keys)
    else:
        raise RuntimeError(
            f"unsupported AI_TRADING_SECRETS_BACKEND '{backend}' "
            "(supported: none, aws-secrets-manager)"
        )

    out_entries: list[EnvEntry] = []
    seen: set[str] = set()
    manager_overrides_applied = 0
    for entry in entries:
        key = entry.key
        if key in DEPRECATED_KEYS:
            continue
        value = entry.value
        if key in managed_keys and backend not in _BACKEND_NONE:
            managed_value = manager_values.get(key)
            if managed_value not in (None, ""):
                value = managed_value
                manager_overrides_applied += 1
            elif require_managed:
                raise RuntimeError(
                    f"managed secret key '{key}' missing in secrets backend payload"
                )
        out_entries.append(EnvEntry(key=key, value=value))
        seen.add(key)

    if backend not in _BACKEND_NONE:
        for key in sorted(managed_keys):
            if key in seen:
                continue
            managed_value = manager_values.get(key)
            if managed_value in (None, ""):
                if require_managed:
                    raise RuntimeError(
                        f"managed secret key '{key}' missing in secrets backend payload"
                    )
                continue
            out_entries.append(EnvEntry(key=key, value=managed_value))
            seen.add(key)
            manager_overrides_applied += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(f"{entry.key}={entry.value}" for entry in out_entries)
    if rendered:
        rendered = f"{rendered}\n"
    dst.write_text(rendered, encoding="utf-8")
    os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)

    return {
        "src": str(src),
        "dst": str(dst),
        "line_count": len(out_entries),
        "managed_key_count": len(managed_keys),
        "manager_overrides_applied": manager_overrides_applied,
        "secrets_backend": backend or "none",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render .env.runtime from .env with secrets overlay.")
    parser.add_argument("--src", default=".env", help="Source env file")
    parser.add_argument("--dst", default=".env.runtime", help="Destination runtime env file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary to stdout",
    )
    args = parser.parse_args(argv)

    summary = _render_runtime_env(
        src=Path(args.src).expanduser().resolve(),
        dst=Path(args.dst).expanduser().resolve(),
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
