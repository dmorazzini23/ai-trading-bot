"""Prediction executor utilities."""

from __future__ import annotations

from typing import Iterable

from ai_trading.config.management import get_env

# Supported trading environments
_SUPPORTED_ENVS: tuple[str, ...] = ("development", "paper", "live", "backtest")


def _validate_env(env: str, allowed: Iterable[str] = _SUPPORTED_ENVS) -> str:
    """Validate trading environment value.

    Parameters
    ----------
    env:
        Environment string to validate.
    allowed:
        Iterable of supported environment names.

    Returns
    -------
    str
        The validated environment name.

    Raises
    ------
    ValueError
        If ``env`` is not in ``allowed``.
    """
    if env not in allowed:
        raise ValueError(f"Unsupported TRADING_ENV: {env!r}")
    return env


def run() -> str:
    """Entry point for prediction execution.

    Reads ``TRADING_ENV`` from the process environment and ensures it is
    supported. The value is returned for downstream use.
    """
    env = get_env("TRADING_ENV", "development", cast=str, resolve_aliases=False)
    return _validate_env(env)


__all__ = ["run"]
