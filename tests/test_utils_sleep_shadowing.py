from __future__ import annotations

import importlib


def test_sleep_uses_stdlib(monkeypatch):
    utils = importlib.import_module("ai_trading.utils")

    slept = {"count": 0}

    def fake_sleep(_):
        slept["count"] += 1

    real_time = importlib.import_module("time")
    monkeypatch.setattr(real_time, "sleep", fake_sleep)

    utils.psleep(0)
    utils.sleep_s(0)
    utils.sleep(0)

    assert slept["count"] == 3

