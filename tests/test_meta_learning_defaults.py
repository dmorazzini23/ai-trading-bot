import pytest


def test_load_global_signal_performance_defaults(monkeypatch):
    """load_global_signal_performance uses relaxed defaults for activation."""
    pd = pytest.importorskip("pandas")
    from ai_trading.core import bot_engine

    # Ensure environment does not override defaults
    monkeypatch.delenv("METALEARN_MIN_TRADES", raising=False)
    monkeypatch.delenv("METALEARN_PERFORMANCE_THRESHOLD", raising=False)

    # Data: alpha has 1 win/2 losses -> 0.333 > 0.3; beta has 2 losses -> 0; delta has 2 wins -> 1.0
    df = pd.DataFrame([
        {"exit_price": 110, "entry_price": 100, "signal_tags": "alpha", "side": "buy"},
        {"exit_price": 90, "entry_price": 100, "signal_tags": "alpha", "side": "buy"},
        {"exit_price": 95, "entry_price": 100, "signal_tags": "alpha", "side": "buy"},
        {"exit_price": 190, "entry_price": 200, "signal_tags": "beta", "side": "buy"},
        {"exit_price": 195, "entry_price": 200, "signal_tags": "beta", "side": "buy"},
        {"exit_price": 105, "entry_price": 100, "signal_tags": "delta", "side": "buy"},
        {"exit_price": 110, "entry_price": 100, "signal_tags": "delta", "side": "buy"},
    ])

    monkeypatch.setattr(bot_engine, "_read_trade_log", lambda *a, **k: df)

    result = bot_engine.load_global_signal_performance()
    assert result == {
        "alpha": pytest.approx(1 / 3, rel=1e-3),
        "delta": 1.0,
    }
