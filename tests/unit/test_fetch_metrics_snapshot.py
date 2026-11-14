from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fm():
    mod = importlib.import_module("ai_trading.data.fetch.metrics")
    mod.reset()
    yield mod
    mod.reset()






def test_snapshot_unauthorized(fm):
    fm.inc_unauthorized_sip("sip")
    out = fm.snapshot()
    assert out["unauthorized"] == 1
    assert out["empty_payload"] == 0


def test_snapshot_empty_payload(fm):
    fm.inc_empty_payload("AAPL", "1Min")
    out = fm.snapshot()
    assert out["empty_payload"] == 1
    assert out["rate_limit"] == 0
