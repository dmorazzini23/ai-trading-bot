def test_safe_mode_guard_blocks_paper_sampling_safe_mode(monkeypatch):
    import ai_trading.execution.live_trading as live

    monkeypatch.delenv("AI_TRADING_HALT", raising=False)
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_SAFE_MODE_ALLOW_PAPER", "1")
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_ENABLED", "1")
    monkeypatch.setattr(live, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(live, "safe_mode_reason", lambda: "minute_gap")
    monkeypatch.setattr(live.provider_monitor, "is_disabled", lambda provider: False)
    monkeypatch.setattr(live, "_safe_mode_policy", lambda: (True, "paper"))

    blocked = live._safe_mode_guard(symbol="AAPL", side="buy", quantity=5)

    assert blocked is True
