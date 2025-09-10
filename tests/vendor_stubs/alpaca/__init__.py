"""Minimal vendor stub for alpaca-py package used in tests."""

from . import trading, data, common  # re-export subpackages


class APIError(Exception):
    """Generic API error stub."""

    pass


__all__ = ["APIError", "trading", "data", "common"]
