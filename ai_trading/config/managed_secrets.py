"""Runtime hydration for externally managed secret values."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.managed_secrets_transport import (
    AWS_CLI_PATH_KEYS,
    AWS_ENV_ALLOWLIST,
    BACKEND_AWS,
    BACKEND_NONE,
    aws_cli_env as _transport_aws_cli_env,
    fetch_aws_secret_payload as _transport_fetch_aws_secret_payload,
    parse_secret_string,
    resolve_aws_cli_path as _transport_resolve_aws_cli_path,
)

logger = get_logger(__name__)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEY_HINT_RE = re.compile(r"(SECRET|TOKEN|PASSWORD|WEBHOOK_URL$|API_KEY$)")


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


def _transport_env() -> dict[str, str]:
    """Resolve only the values required by the AWS CLI subprocess."""

    keys = AWS_ENV_ALLOWLIST | AWS_CLI_PATH_KEYS
    return {
        key: str(value)
        for key in keys
        if (value := get_env(key, None, cast=str, resolve_aliases=False)) is not None
    }


def resolve_aws_cli_path() -> str:
    return _transport_resolve_aws_cli_path(env=_transport_env())


def aws_cli_env() -> dict[str, str]:
    return _transport_aws_cli_env(env=_transport_env())


def fetch_aws_secret_payload(secret_id: str, *, region: str, profile: str) -> dict[str, str]:
    return _transport_fetch_aws_secret_payload(
        secret_id,
        region=region,
        profile=profile,
        env=_transport_env(),
    )


def hydrate_managed_secrets(*, required_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Load managed secrets into process-local config overrides without writing files."""

    from ai_trading.config.management import set_runtime_env_override

    backend = str(get_env("AI_TRADING_SECRETS_BACKEND", "none", resolve_aliases=False) or "none").lower()
    if backend in BACKEND_NONE:
        return {"secrets_backend": backend or "none", "hydrated_count": 0}
    if backend not in BACKEND_AWS:
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

    if {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"} & set(hydrated):
        from ai_trading.utils.env import refresh_alpaca_credentials_cache

        refresh_alpaca_credentials_cache()
        data_fetch = sys.modules.get("ai_trading.data.fetch")
        refresh_default_feed = getattr(data_fetch, "refresh_default_feed", None)
        if callable(refresh_default_feed):
            refresh_default_feed()

    logger.info(
        "MANAGED_SECRETS_HYDRATED",
        extra={"secrets_backend": backend, "hydrated_keys": tuple(hydrated)},
    )
    return {
        "secrets_backend": backend,
        "hydrated_count": len(hydrated),
        "hydrated_keys": tuple(hydrated),
    }


_BACKEND_NONE = BACKEND_NONE
_BACKEND_AWS = BACKEND_AWS
_parse_secret_string = parse_secret_string
_resolve_aws_cli_path = resolve_aws_cli_path
_aws_cli_env = aws_cli_env
_fetch_aws_secret_payload = fetch_aws_secret_payload


__all__ = [
    "BACKEND_AWS",
    "BACKEND_NONE",
    "aws_cli_env",
    "fetch_aws_secret_payload",
    "hydrate_managed_secrets",
    "parse_secret_string",
    "resolve_aws_cli_path",
]
