"""Runtime hydration for externally managed secret values."""

from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEY_HINT_RE = re.compile(r"(SECRET|TOKEN|PASSWORD|WEBHOOK_URL$|API_KEY$)")
_BACKEND_NONE = {"", "none", "off", "disabled"}
_BACKEND_AWS = {"aws-secrets-manager", "aws_sm", "aws"}
_AWS_ENV_ALLOWLIST = {
    "AWS_ACCESS_KEY_ID",
    "AWS_CA_BUNDLE",
    "AWS_CONFIG_FILE",
    "AWS_DEFAULT_REGION",
    "AWS_EC2_METADATA_DISABLED",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_SDK_LOAD_CONFIG",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "PATH",
}


def _parse_bool(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv_keys(raw: Any) -> set[str]:
    if not raw:
        return set()
    keys: set[str] = set()
    for part in str(raw).split(","):
        key = part.strip().upper()
        if _ENV_KEY_RE.match(key):
            keys.add(key)
    return keys


def _secret_like_keys(keys: Iterable[str]) -> set[str]:
    return {
        str(key).strip().upper()
        for key in keys
        if _SECRET_KEY_HINT_RE.search(str(key).strip().upper())
    }


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
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key_raw, value_raw = raw_line.split("=", 1)
        key = key_raw.strip().upper()
        if _ENV_KEY_RE.match(key):
            result[key] = value_raw.strip()
    return result


def _resolve_aws_cli_path() -> str:
    configured = str(os.getenv("AI_TRADING_AWS_CLI_PATH") or os.getenv("AWS_CLI_PATH") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise RuntimeError("AI_TRADING_AWS_CLI_PATH must be an absolute path")
        if not path.exists():
            raise RuntimeError(f"configured aws CLI path does not exist: {path}")
        return str(path)
    return shutil.which("aws") or "aws"


def _aws_cli_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _AWS_ENV_ALLOWLIST}
    candidate_homes: list[Path] = []
    home = str(env.get("HOME") or "").strip()
    if home:
        candidate_homes.append(Path(home))
    try:
        passwd_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
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


def _fetch_aws_secret_payload(secret_id: str, *, region: str, profile: str) -> dict[str, str]:
    command: list[str] = [_resolve_aws_cli_path()]
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
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=_aws_cli_env(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("aws CLI not found. Install AWS CLI v2 and ensure `aws` is on PATH.") from exc
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
            f"AWS secret '{secret_id}' did not contain key/value entries"
        )
    return parsed


def hydrate_managed_secrets(*, required_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Load managed secrets into process-local config overrides without writing files."""

    from ai_trading.config.management import get_env, set_runtime_env_override

    backend = str(get_env("AI_TRADING_SECRETS_BACKEND", "none", resolve_aliases=False) or "none").lower()
    if backend in _BACKEND_NONE:
        return {"secrets_backend": backend or "none", "hydrated_count": 0}
    if backend not in _BACKEND_AWS:
        raise RuntimeError(
            f"unsupported AI_TRADING_SECRETS_BACKEND '{backend}' "
            "(supported: none, aws-secrets-manager)"
        )

    secret_id = str(get_env("AI_TRADING_AWS_SECRET_ID", "", resolve_aliases=False) or "").strip()
    if not secret_id:
        raise RuntimeError(
            "AI_TRADING_SECRETS_BACKEND is enabled but AI_TRADING_AWS_SECRET_ID is empty"
        )
    region = str(get_env("AI_TRADING_AWS_REGION", "", resolve_aliases=False) or "").strip()
    profile = str(get_env("AI_TRADING_AWS_PROFILE", "", resolve_aliases=False) or "").strip()
    require_managed = _parse_bool(
        get_env("AI_TRADING_REQUIRE_MANAGED_SECRETS", "0", resolve_aliases=False),
        default=False,
    )
    explicit_keys = _parse_csv_keys(
        get_env("AI_TRADING_MANAGED_SECRET_KEYS", "", resolve_aliases=False)
    )
    excluded_keys = _parse_csv_keys(
        get_env("AI_TRADING_EXCLUDED_MANAGED_SECRET_KEYS", "", resolve_aliases=False)
    )
    payload = _fetch_aws_secret_payload(secret_id, region=region, profile=profile)
    required_secret_keys = _secret_like_keys(required_keys)
    keys_to_hydrate = (explicit_keys | required_secret_keys) - excluded_keys
    if not keys_to_hydrate:
        keys_to_hydrate = _secret_like_keys(payload)

    missing = sorted(
        key for key in keys_to_hydrate if payload.get(key) in (None, "")
    )
    if missing and require_managed:
        raise RuntimeError(
            "managed secret keys missing in secrets backend payload: "
            + ", ".join(missing)
        )

    hydrated: list[str] = []
    for key in sorted(keys_to_hydrate):
        value = payload.get(key)
        if value in (None, ""):
            continue
        set_runtime_env_override(key, value)
        hydrated.append(key)

    logger.info(
        "MANAGED_SECRETS_HYDRATED",
        extra={"secrets_backend": backend, "hydrated_keys": tuple(hydrated)},
    )
    return {
        "secrets_backend": backend,
        "hydrated_count": len(hydrated),
        "hydrated_keys": tuple(hydrated),
    }


__all__ = ["hydrate_managed_secrets"]
