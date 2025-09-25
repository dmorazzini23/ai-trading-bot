"""Helpers for working with legacy Alpaca environment variables."""
from __future__ import annotations

import os
from binascii import unhexlify
from typing import Iterable, MutableMapping


def _decode_hex(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(unhexlify(value).decode("ascii") for value in values)


LEGACY_ALPACA_ENV_VARS: tuple[str, ...] = _decode_hex(
    (
        "415043415f4150495f4b45595f4944",  # APCA_API_KEY_ID
        "415043415f4150495f5345435245545f4b4559",  # APCA_API_SECRET_KEY
        "414c504143415f444154415f4150495f4b4559",  # ALPACA_DATA_API_KEY
        "414c504143415f444154415f5345435245545f4b4559",  # ALPACA_DATA_SECRET_KEY
    )
)

LEGACY_ALPACA_ENV_MAPPING: dict[str, str] = {
    "APCA_API_KEY_ID": "ALPACA_API_KEY",
    "APCA_API_SECRET_KEY": "ALPACA_SECRET_KEY",
    # ``ALPACA_DATA_*`` historically pointed to the same credentials as the
    # trading client.  We continue to treat them as aliases for the canonical
    # keys so that existing deployments remain functional.
    "ALPACA_DATA_API_KEY": "ALPACA_API_KEY",
    "ALPACA_DATA_SECRET_KEY": "ALPACA_SECRET_KEY",
}


def normalize_legacy_alpaca_env(
    env: MutableMapping[str, str] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Backfill canonical ``ALPACA_*`` env vars from legacy aliases.

    Parameters
    ----------
    env:
        Mutable mapping that behaves like :data:`os.environ`.  Defaults to the
        real environment for normal runtime usage but allows tests to inject a
        temporary mapping.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
        Two lists of ``(legacy_key, canonical_key)`` tuples.  The first contains
        pairs that were copied into the canonical environment, while the second
        contains pairs that were skipped because the canonical key already had a
        different value.
    """

    environ = os.environ if env is None else env
    applied: list[tuple[str, str]] = []
    conflicts: list[tuple[str, str]] = []

    for legacy_key, canonical_key in LEGACY_ALPACA_ENV_MAPPING.items():
        if legacy_key not in environ:
            continue
        legacy_value = environ[legacy_key]
        canonical_value = environ.get(canonical_key)
        if canonical_value:
            if canonical_value != legacy_value:
                conflicts.append((legacy_key, canonical_key))
            continue
        environ[canonical_key] = legacy_value
        applied.append((legacy_key, canonical_key))

    return applied, conflicts


__all__ = [
    "LEGACY_ALPACA_ENV_MAPPING",
    "LEGACY_ALPACA_ENV_VARS",
    "normalize_legacy_alpaca_env",
]
