"""Import-safe transport helpers for externally managed secrets.

This module deliberately does not import runtime configuration.  Operational
bootstrap scripts can therefore use it before the canonical runtime env file
has been rendered.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BACKEND_NONE = {"", "none", "off", "disabled"}
BACKEND_AWS = {"aws-secrets-manager", "aws_sm", "aws"}
AWS_ENV_ALLOWLIST = {
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
AWS_CLI_PATH_KEYS = {"AI_TRADING_AWS_CLI_PATH", "AWS_CLI_PATH"}


def parse_secret_string(raw: str) -> dict[str, str]:
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


def resolve_aws_cli_path(*, env: Mapping[str, str]) -> str:
    configured = str(
        env.get("AI_TRADING_AWS_CLI_PATH", "")
        or env.get("AWS_CLI_PATH", "")
        or ""
    ).strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise RuntimeError("AI_TRADING_AWS_CLI_PATH must be an absolute path")
        if not path.exists():
            raise RuntimeError(f"configured aws CLI path does not exist: {path}")
        return str(path)
    return shutil.which("aws", path=env.get("PATH")) or "aws"


def aws_cli_env(*, env: Mapping[str, str]) -> dict[str, str]:
    cli_env = {
        key: str(value)
        for key in AWS_ENV_ALLOWLIST
        if (value := env.get(key)) is not None
    }
    candidate_homes: list[Path] = []
    home = str(cli_env.get("HOME") or "").strip()
    if home:
        candidate_homes.append(Path(home))
    try:
        passwd_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        passwd_home = None
    if passwd_home is not None and passwd_home not in candidate_homes:
        candidate_homes.append(passwd_home)
    if not home and candidate_homes:
        cli_env["HOME"] = str(candidate_homes[0])

    creds_already = str(cli_env.get("AWS_SHARED_CREDENTIALS_FILE") or "").strip()
    config_already = str(cli_env.get("AWS_CONFIG_FILE") or "").strip()
    for base in candidate_homes:
        aws_dir = base / ".aws"
        creds_file = aws_dir / "credentials"
        config_file = aws_dir / "config"
        if not creds_already and creds_file.exists():
            cli_env["AWS_SHARED_CREDENTIALS_FILE"] = str(creds_file)
            creds_already = cli_env["AWS_SHARED_CREDENTIALS_FILE"]
        if not config_already and config_file.exists():
            cli_env["AWS_CONFIG_FILE"] = str(config_file)
            config_already = cli_env["AWS_CONFIG_FILE"]
        if creds_already and config_already:
            break
    return cli_env


def fetch_aws_secret_payload(
    secret_id: str,
    *,
    region: str,
    profile: str,
    env: Mapping[str, str],
) -> dict[str, str]:
    command: list[str] = [resolve_aws_cli_path(env=env)]
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
            env=aws_cli_env(env=env),
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
    parsed = parse_secret_string(secret_string)
    if not parsed:
        raise RuntimeError(f"AWS secret '{secret_id}' did not contain key/value entries")
    return parsed


__all__ = [
    "AWS_CLI_PATH_KEYS",
    "AWS_ENV_ALLOWLIST",
    "BACKEND_AWS",
    "BACKEND_NONE",
    "aws_cli_env",
    "fetch_aws_secret_payload",
    "parse_secret_string",
    "resolve_aws_cli_path",
]
