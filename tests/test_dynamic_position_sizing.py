from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import ai_trading.position_sizing as ps


class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):  # pragma: no cover - simple
        return self._payload


def _stub_session(monkeypatch, status: int, payload: dict, *, calls: dict | None = None) -> None:
    class Sess:
        def get(self, url, headers=None):
            if calls is not None:
                calls["n"] = calls.get("n", 0) + 1
            return _Resp(status, payload)

    monkeypatch.setattr(ps, "get_global_session", lambda: Sess())


def test_auto_mode_resolves_from_equity_and_capital_cap(monkeypatch, caplog):
    ps._CACHE.value, ps._CACHE.ts, ps._CACHE.equity, ps._CACHE.equity_error = (None, None, None, None)
    cfg = SimpleNamespace(
        alpaca_base_url="https://paper-api.alpaca.markets",
        alpaca_api_key="k",
        alpaca_secret_key_plain="s",
        max_position_mode="AUTO",
    )
    tcfg = SimpleNamespace(
        capital_cap=0.04,
        max_position_mode="AUTO",
        dynamic_size_refresh_secs=3600,
    )

    _stub_session(monkeypatch, 200, {"equity": "100000.00"})

    with caplog.at_level(logging.INFO):
        size, meta = ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)

    assert size == 4000.0
    assert meta["source"] in {"alpaca", "cache"}


def test_auto_mode_raises_on_error_without_cached_equity(monkeypatch, caplog):
    ps._CACHE.value, ps._CACHE.ts, ps._CACHE.equity, ps._CACHE.equity_error = (None, None, None, None)
    cfg = SimpleNamespace(
        alpaca_base_url="https://paper-api.alpaca.markets",
        alpaca_api_key="k",
        alpaca_secret_key_plain="s",
        max_position_mode="AUTO",
        max_position_equity_fallback=100000.0,
    )
    tcfg = SimpleNamespace(
        capital_cap=0.04,
        max_position_mode="AUTO",
        dynamic_size_refresh_secs=3600,
    )

    calls: dict[str, int] = {}
    _stub_session(monkeypatch, 500, {}, calls=calls)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="http_error:500"):
            ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)

    abort_logs = [r for r in caplog.records if r.msg == "AUTO_SIZING_ABORTED"]
    assert abort_logs and getattr(abort_logs[0], "reason", None) == "http_error:500"
    assert calls["n"] == 1

