"""Helpers for generating stable identifiers used across execution paths."""

from __future__ import annotations

import secrets
import string

_SUFFIX_ALPHABET = string.ascii_lowercase + string.digits


def _random_suffix(length: int = 8) -> str:
    return "".join(secrets.choice(_SUFFIX_ALPHABET) for _ in range(length))


def stable_client_order_id(symbol: str, side: str, epoch_min: int) -> str:
    """Return a client order identifier derived from the trade context."""

    base = f"{symbol}-{side}-{int(epoch_min)}"
    return f"{base}-{_random_suffix()}"


__all__ = ["stable_client_order_id"]
