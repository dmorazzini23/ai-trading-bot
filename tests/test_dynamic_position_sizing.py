from __future__ import annotations

import logging
from types import SimpleNamespace

from ai_trading.net.http import get_global_session
from ai_trading.position_sizing import resolve_max_position_size


class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _stub_session(monkeypatch, status: int, payload: dict) -> None:
    sess = get_global_session()

    def fake_get(url, headers=None):  # noqa: D401
        return _Resp(status, payload)

    monkeypatch.setattr(sess, "get", fake_get, raising=True)


def test_auto_mode_resolves_from_equity_and_capital_cap(monkeypatch, caplog):
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
        size, meta = resolve_max_position_size(cfg, tcfg, force_refresh=True)

    assert size == 4000.0
    assert meta["source"] in {"alpaca", "cache"}


def test_auto_mode_fallback_on_error(monkeypatch, caplog):
    cfg = SimpleNamespace(
        alpaca_base_url="https://paper-api.alpaca.markets",
        alpaca_api_key="k",
        alpaca_secret_key_plain="s",
        default_max_position_size=9000.0,
        max_position_mode="AUTO",
    )
    tcfg = SimpleNamespace(
        capital_cap=0.04,
        max_position_mode="AUTO",
        dynamic_size_refresh_secs=3600,
    )

    _stub_session(monkeypatch, 500, {})

    with caplog.at_level(logging.WARNING):
        size, meta = resolve_max_position_size(cfg, tcfg, force_refresh=True)

    assert size == 9000.0
    assert meta["source"] == "fallback"

