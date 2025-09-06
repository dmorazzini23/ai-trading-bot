from __future__ import annotations

import importlib

import pytest

import ai_trading.data.fetch as df
from ai_trading.data.metrics import metrics as metrics_state


@pytest.fixture(autouse=True)
def reset_counters():
    metrics_state.rate_limit = 0
    metrics_state.timeout = 0
    metrics_state.unauthorized = 0
    metrics_state.empty_payload = 0
    metrics_state.feed_switch = 0
    yield
    metrics_state.rate_limit = 0
    metrics_state.timeout = 0
    metrics_state.unauthorized = 0
    metrics_state.empty_payload = 0
    metrics_state.feed_switch = 0


@pytest.fixture
def fm():
    mod = importlib.import_module("ai_trading.data.fetch.metrics")
    # Ensure other tests still see the dataclass instance
    df.metrics = metrics_state
    return mod


def test_snapshot_rate_limit(fm):
    metrics_state.rate_limit += 1
    out = fm.snapshot(None)
    assert out["rate_limit"] == 1
    assert out["timeout"] == 0


def test_snapshot_timeout(fm):
    metrics_state.timeout += 2
    out = fm.snapshot(object())
    assert out["timeout"] == 2
    assert out["rate_limit"] == 0


def test_snapshot_unauthorized(fm):
    metrics_state.unauthorized += 1
    out = fm.snapshot(None)
    assert out["unauthorized"] == 1
    assert out["empty_payload"] == 0


def test_snapshot_empty_payload(fm):
    metrics_state.empty_payload += 3
    out = fm.snapshot(None)
    assert out["empty_payload"] == 3
    assert out["rate_limit"] == 0
