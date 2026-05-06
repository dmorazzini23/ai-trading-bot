#!/usr/bin/env python3
"""Render a runtime env file from `.env`, verifying managed secrets."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ai_trading.config.managed_secrets import (
    BACKEND_AWS,
    BACKEND_NONE,
    aws_cli_env as _aws_cli_env,
    fetch_aws_secret_payload as _fetch_aws_secret_payload,
    parse_secret_string as _parse_secret_string,
)

try:
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - script can fall back before deps are installed
    dotenv_values = None

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
    "AI_TRADING_GRAFANA_API_TOKEN",
    "AI_TRADING_PROM_REMOTE_WRITE_PASSWORD",
    "TRADIER_ACCESS_TOKEN",
    "FINNHUB_API_KEY",
    "SENTIMENT_API_KEY",
    "IEX_API_TOKEN",
    "FRED_API_KEY",
}

_DEFAULT_EXCLUDED_MANAGED_KEYS = {
    "AI_TRADING_AWS_PROFILE",
    "AI_TRADING_AWS_REGION",
    "AI_TRADING_AWS_SECRET_ID",
    "AI_TRADING_EXCLUDED_MANAGED_SECRET_KEYS",
    "AI_TRADING_MANAGED_SECRET_KEYS",
    "AI_TRADING_REQUIRE_MANAGED_SECRETS",
    "AI_TRADING_SECRETS_BACKEND",
    "AWS_CONFIG_FILE",
    "AWS_REGION",
    "AWS_SHARED_CREDENTIALS_FILE",
}

_BACKEND_NONE = BACKEND_NONE
_BACKEND_AWS = BACKEND_AWS


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
    if dotenv_values is not None:
        parsed = dotenv_values(path)
        return [
            EnvEntry(key=str(key), value=str(value))
            for key, value in parsed.items()
            if value is not None and _ENV_KEY_RE.match(str(key))
        ]
    return _parse_env_entries(path.read_text(encoding="utf-8", errors="ignore"))


def _entries_to_map(entries: list[EnvEntry]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        result[entry.key] = entry.value
    return result


_SAFE_UNQUOTED_VALUE_RE = re.compile(r"^[A-Za-z0-9_@%+=:,./-]*$")


def _quote_env_value(value: str) -> str:
    """Render a dotenv/systemd EnvironmentFile-compatible value."""

    if value and _SAFE_UNQUOTED_VALUE_RE.match(value):
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


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


def _excluded_managed_keys(*, env_map: Mapping[str, str]) -> set[str]:
    return _DEFAULT_EXCLUDED_MANAGED_KEYS | _parse_csv_keys(
        _resolve_setting("AI_TRADING_EXCLUDED_MANAGED_SECRET_KEYS", env_map=env_map)
    )


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


def _write_runtime_env_atomic(dst: Path, rendered: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    temp_path = Path(temp_name)
    try:
        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, dst)
        try:
            dir_fd = os.open(dst.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


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
    excluded_managed_keys = _excluded_managed_keys(env_map=env_map)
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
    managed_keys.difference_update(excluded_managed_keys)

    out_entries: list[EnvEntry] = []
    seen: set[str] = set()
    manager_overrides_applied = 0
    managed_keys_verified = 0
    managed_secret_values_omitted = 0
    managed_secret_values_written = 0
    for entry in entries:
        key = entry.key
        if key in DEPRECATED_KEYS:
            continue
        value = entry.value
        if key in managed_keys and backend not in _BACKEND_NONE:
            managed_value = manager_values.get(key)
            if managed_value not in (None, ""):
                managed_keys_verified += 1
                manager_overrides_applied += 1
                managed_secret_values_omitted += 1
                out_entries.append(EnvEntry(key=key, value=""))
            elif require_managed:
                raise RuntimeError(
                    f"managed secret key '{key}' missing in secrets backend payload"
                )
            else:
                managed_secret_values_omitted += 1
                out_entries.append(EnvEntry(key=key, value=""))
            seen.add(key)
            continue
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
                managed_secret_values_omitted += 1
                continue
            managed_keys_verified += 1
            managed_secret_values_omitted += 1
            seen.add(key)

    rendered = "\n".join(
        f"{entry.key}={_quote_env_value(entry.value)}"
        for entry in out_entries
    )
    if rendered:
        rendered = f"{rendered}\n"
    _write_runtime_env_atomic(dst, rendered)

    return {
        "src": str(src),
        "dst": str(dst),
        "line_count": len(out_entries),
        "managed_key_count": len(managed_keys),
        "manager_overrides_applied": manager_overrides_applied,
        "managed_keys_verified": managed_keys_verified,
        "managed_secret_values_omitted": managed_secret_values_omitted,
        "managed_secret_values_written": managed_secret_values_written,
        "secrets_backend": backend or "none",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render runtime env from .env with secrets overlay.")
    parser.add_argument("--src", default=".env", help="Source env file")
    parser.add_argument("--dst", default="runtime/ai-trading-runtime.env", help="Destination runtime env file")
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
