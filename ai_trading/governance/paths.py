from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from pathlib import Path

from ai_trading.config.management import get_env


def resolve_governance_base_path(configured: str | None = None) -> Path:
    """Return the canonical governance artifact directory.

    Prefer an explicit governance path when configured. Otherwise default to a
    writable runtime location under the configured output directory.
    """

    configured_token = str(configured or "").strip()
    if not configured_token:
        try:
            configured_token = str(
                get_env(
                    "AI_TRADING_GOVERNANCE_BASE_PATH",
                    "",
                    cast=str,
                    resolve_aliases=False,
                )
                or ""
            ).strip()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            configured_token = ""
    if configured_token:
        resolved = Path(configured_token).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        return resolved

    try:
        output_dir = str(
            get_env(
                "AI_TRADING_OUTPUT_DIR",
                "runtime",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime"
        ).strip() or "runtime"
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        output_dir = "runtime"
    base = Path(output_dir).expanduser()
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    return (base / "governance").resolve()


__all__ = ["resolve_governance_base_path"]
