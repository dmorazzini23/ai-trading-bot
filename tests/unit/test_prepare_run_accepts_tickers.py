from __future__ import annotations

import types

import ai_trading.core.bot_engine as eng


# ensure _prepare_run uses provided tickers without raising TypeError

def test_prepare_run_accepts_ticker_list(monkeypatch):
    runtime = types.SimpleNamespace(params={}, cfg=None)
    state = eng.BotState()
    tickers = ["AAPL", "MSFT"]

    # Stub functions called within _prepare_run
    monkeypatch.setattr(eng, "ensure_data_fetcher", lambda rt: None)
    monkeypatch.setattr(eng, "cancel_all_open_orders", lambda rt: None)
    monkeypatch.setattr(eng, "audit_positions", lambda rt: None)
    mock_acct = types.SimpleNamespace(equity="1000", buying_power="1000", cash="1000")
    monkeypatch.setattr(eng, "safe_alpaca_get_account", lambda rt: mock_acct)
    monkeypatch.setattr(eng, "compute_spy_vol_stats", lambda rt: None)
    monkeypatch.setattr(eng, "pretrade_data_health", lambda rt, syms: None)
    monkeypatch.setattr(eng, "screen_candidates", lambda rt, syms: syms)
    monkeypatch.setattr(eng, "pre_trade_health_check", lambda rt, syms: {})
    monkeypatch.setattr(eng, "check_market_regime", lambda rt, st: True)
    monkeypatch.setattr(eng, "_param", lambda rt, key, default: default)
    monkeypatch.setattr(eng.portfolio, "compute_portfolio_weights", lambda rt, syms: {})

    class DummyLock:
        def __enter__(self):  # pragma: no cover - simple stub
            return None

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple stub
            return False

    monkeypatch.setattr(eng, "portfolio_lock", DummyLock())

    current_cash, regime_ok, symbols = eng._prepare_run(runtime, state, tickers)

    assert current_cash == 1000.0
    assert regime_ok is True
    assert symbols == tickers
