"""Read-only check of Alpaca account configurations against the launch profile.

Usage::

    python -m ai_trading.tools.broker_account_configuration --check [--env-file PATH]

Prints observed vs expected configuration and the mismatch list as JSON.
Exit codes: 0 when matching, 2 on drift, 3 when the configuration could not be
read.  There is deliberately no ``--apply``; to change the account, follow
``docs/BROKER_ACCOUNT_CONFIGURATION.md`` (the exact SDK call is also in the
``ai_trading.broker.account_configuration`` module docstring).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ai_trading.broker.account_configuration import check_account_configuration
from ai_trading.broker.alpaca_credentials import (
    initialize,
    resolve_alpaca_credentials_with_base,
)
from ai_trading.config.managed_secrets import hydrate_managed_secrets
from ai_trading.config.management import reload_env
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

EXIT_MATCH = 0
EXIT_DRIFT = 2
EXIT_UNAVAILABLE = 3


def _build_client() -> tuple[Any | None, dict[str, Any]]:
    context: dict[str, Any] = {}
    try:
        hydrate_managed_secrets(required_keys=("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
        creds = resolve_alpaca_credentials_with_base()
        context["paper"] = "paper" in creds.base_url.lower()
        context["base_url"] = creds.base_url
        if not creds.api_key or not creds.secret_key:
            context["client_error"] = "alpaca_credentials_missing"
            return None, context
        return initialize(), context
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        context["client_error"] = f"{type(exc).__name__}: {exc}"[:240]
        return None, context


def run_check(client: Any | None = None) -> tuple[int, dict[str, Any]]:
    context: dict[str, Any] = {}
    if client is None:
        client, context = _build_client()
    result = check_account_configuration(client)
    if client is None and context.get("client_error"):
        result["reason"] = context["client_error"]
    payload = {**context, **result}
    payload.pop("client_error", None)
    if not result["available"]:
        return EXIT_UNAVAILABLE, payload
    return (EXIT_DRIFT if result["mismatches"] else EXIT_MATCH), payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only comparison of Alpaca account configurations with the "
            "active launch profile."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        required=True,
        help="Fetch account configurations and compare them (read-only).",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Dotenv file to load credentials and profile settings from "
        "(default: ./.env when present).",
    )
    args = parser.parse_args(argv)
    reload_env(args.env_file)
    code, payload = run_check()
    sys.stdout.write(f"{json.dumps(payload, indent=2, sort_keys=True, default=str)}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
