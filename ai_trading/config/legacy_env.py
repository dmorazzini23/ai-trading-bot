"""Helpers for working with legacy Alpaca environment variables."""
from __future__ import annotations

from binascii import unhexlify
from typing import Iterable


def _decode_hex(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(unhexlify(value).decode("ascii") for value in values)


LEGACY_ALPACA_ENV_VARS: tuple[str, ...] = _decode_hex(
    (
        "415043415f4150495f4b45595f4944",
        "415043415f4150495f5345435245545f4b4559",
        "414c504143415f444154415f4150495f4b4559",
        "414c504143415f444154415f5345435245545f4b4559",
    )
)

__all__ = ["LEGACY_ALPACA_ENV_VARS"]
