from __future__ import annotations

import types
from unittest.mock import Mock

import ai_trading.core.bot_engine as eng
from ai_trading.data.dynamic_universe import DynamicUniverseResult, UniverseCandidate


def test_prepare_run_maintains_base_and_cycle_universes(monkeypatch):
    runtime = types.SimpleNamespace(params={}, cfg=None)
    state = eng.BotState()
    base = ["AAPL", "MSFT"]
    overlay = DynamicUniverseResult(
        merged_symbols=["NVDA", "AAPL", "MSFT"],
        additions=[
            UniverseCandidate(
                symbol="NVDA",
                source="alpaca_gainer",
                side_bias="long",
                rank=1,
                pct_change=5.1,
                last_price=900.0,
                volume=1_200_000.0,
                dollar_volume=1_080_000_000.0,
                asof="2026-04-17T15:25:00+00:00",
                reason="accepted",
                tradable=True,
                marginable=True,
                shortable=True,
                easy_to_borrow=True,
            )
        ],
        metadata={"dynamic_symbols": ["NVDA"], "generated_at": "2026-04-17T15:25:00+00:00"},
    )

    monkeypatch.setattr(eng, "ensure_data_fetcher", lambda rt: None)
    monkeypatch.setattr(eng, "cancel_all_open_orders", lambda rt: None)
    monkeypatch.setattr(eng, "audit_positions", lambda rt: None)
    monkeypatch.setattr(
        eng,
        "safe_alpaca_get_account",
        lambda rt: types.SimpleNamespace(equity="1000", buying_power="1000", cash="1000"),
    )
    monkeypatch.setattr(eng, "compute_spy_vol_stats", lambda rt: None)
    monkeypatch.setattr(eng, "pretrade_data_health", lambda rt, syms: None)
    monkeypatch.setattr(eng, "screen_candidates", lambda rt, syms: syms[:2])
    monkeypatch.setattr(eng, "pre_trade_health_check", lambda rt, syms: {})
    monkeypatch.setattr(eng, "check_market_regime", lambda rt, st: True)
    monkeypatch.setattr(eng, "_param", lambda rt, key, default: default)
    monkeypatch.setattr(eng.portfolio, "compute_portfolio_weights", lambda rt, syms: {})
    monkeypatch.setattr(eng, "_resolve_data_provider_degraded", lambda: (False, None, False))
    monkeypatch.setattr(eng, "_degrade_state", lambda snapshot: (False, None, False))
    monkeypatch.setattr(eng, "guard_begin_cycle", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "ai_trading.data.dynamic_universe.build_dynamic_universe",
        lambda runtime, base_universe_tickers: overlay,
    )

    class DummyLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("ai_trading.utils.portfolio_lock", DummyLock(), raising=False)

    _, _, symbols = eng._prepare_run(runtime, state, base)

    assert runtime.base_universe_tickers == base
    assert runtime.universe_tickers == ["NVDA", "AAPL", "MSFT"]
    assert runtime.dynamic_universe_metadata["dynamic_symbols"] == ["NVDA"]
    assert runtime.tickers == ["NVDA", "AAPL"]
    assert symbols == ["NVDA", "AAPL"]


def test_run_all_trades_worker_uses_base_universe_not_previous_cycle_overlay(monkeypatch):
    state = eng.BotState()

    class DummyAPI:
        def get_orders(self, *args, **kwargs):
            return []

        def cancel_order(self, *args, **kwargs):
            return None

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            return None

    runtime = types.SimpleNamespace(
        api=DummyAPI(),
        risk_engine=DummyRiskEngine(),
        model=object(),
        base_universe_tickers=["AAPL", "MSFT"],
        universe_tickers=["NVDA", "AAPL", "MSFT"],
    )
    captured: list[list[str] | None] = []

    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)
    monkeypatch.setattr(eng, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)
    monkeypatch.setattr(eng, "_process_symbols", lambda *args, **kwargs: None)
    monkeypatch.setattr(eng.logger_once, "warning", Mock())

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            return None

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    def fake_prepare(runtime_obj, state_obj, tickers):
        captured.append(list(tickers) if tickers is not None else None)
        return 0.0, True, []

    monkeypatch.setattr(eng, "_prepare_run", fake_prepare)

    eng.run_all_trades_worker(state, runtime)

    assert captured == [["AAPL", "MSFT"]]
