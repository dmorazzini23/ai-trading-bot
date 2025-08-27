"""Test configuration and shared fixtures."""

import os
import sys as _sys

import asyncio
import random
import socket
import sys
from datetime import datetime, timezone
import pathlib

import pytest

try:
    from alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    from alpaca.data import TimeFrame  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - dependency missing
    pytest.skip("alpaca-py is required for tests", allow_module_level=True)

try:
    from freezegun import freeze_time  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    from contextlib import contextmanager

    @contextmanager
    def freeze_time(*_a, **_k):  # AI-AGENT-REF: no-op when freezegun missing
        yield


@pytest.fixture(autouse=True, scope="session")
def _event_loop_policy():
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@pytest.fixture(autouse=True)
def _freeze_clock():
    with freeze_time("2024-01-02 15:04:05", tz_offset=0):
        yield


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(*a, **k):
        raise RuntimeError("Network disabled in tests")

    monkeypatch.setattr(socket, "create_connection", guard, raising=True)


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "dummy")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("CAPITAL_CAP", "0.04")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.05")
    monkeypatch.setenv("MAX_POSITION_SIZE", "1000")


@pytest.fixture(autouse=True)
def _seed_prng() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except Exception:
        pass
    try:
        import torch  # type: ignore
    except Exception:
        pass
    else:
        torch.manual_seed(0)


def reload_module(mod):
    """Reload a module by object or name for tests."""
    import importlib

    if isinstance(mod, str):
        return importlib.reload(importlib.import_module(mod))
    return importlib.reload(mod)
