from __future__ import annotations

import types

import ai_trading.core.bot_engine as eng


def test_prepare_run_cancels_open_orders_only_once(monkeypatch):
    runtime = types.SimpleNamespace(params={}, cfg=None)
    state = eng.BotState()

    cancel_calls: list[int] = []

    def _cancel_stub(_runtime):
        cancel_calls.append(1)

    mock_acct = types.SimpleNamespace(equity="1000", buying_power="1000", cash="1000")
    mock_cfg = types.SimpleNamespace(
        skip_compute_when_provider_disabled=False,
        degraded_feed_mode="widen",
        safe_mode_failsoft=True,
    )

    # Stub dependencies to isolate order-cancel behaviour
    monkeypatch.setattr(eng, "ensure_data_fetcher", lambda rt: None)
    monkeypatch.setattr(eng, "cancel_all_open_orders", _cancel_stub)
    monkeypatch.setattr(eng, "audit_positions", lambda rt: None)
    monkeypatch.setattr(eng, "safe_alpaca_get_account", lambda rt: mock_acct)
    monkeypatch.setattr(eng, "compute_spy_vol_stats", lambda rt: None)
    monkeypatch.setattr(eng, "pretrade_data_health", lambda rt, syms: None)
    monkeypatch.setattr(eng, "screen_candidates", lambda rt, syms: syms)
    monkeypatch.setattr(eng, "pre_trade_health_check", lambda rt, syms: {})
    monkeypatch.setattr(eng, "check_market_regime", lambda rt, st: True)
    monkeypatch.setattr(eng, "_param", lambda rt, key, default: default)
    monkeypatch.setattr(eng, "get_trading_config", lambda: mock_cfg)
    monkeypatch.setattr(eng, "_failsoft_mode_active", lambda *_, **__: False)
    monkeypatch.setattr(eng, "_resolve_data_provider_degraded", lambda: (False, None, False))
    monkeypatch.setattr(eng, "_degrade_state", lambda snapshot: (False, None, False))
    monkeypatch.setattr(eng, "guard_begin_cycle", lambda *_, **__: None)
    monkeypatch.setattr(eng, "load_candidate_universe", lambda rt, syms: syms)
    class _DummyLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("ai_trading.utils.portfolio_lock", _DummyLock(), raising=False)
    monkeypatch.setattr(eng.portfolio, "compute_portfolio_weights", lambda rt, syms: {})

    eng._prepare_run(runtime, state, ["AAPL", "MSFT"])
    eng._prepare_run(runtime, state, ["AAPL", "MSFT"])

    assert len(cancel_calls) == 1
