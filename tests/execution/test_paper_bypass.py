def test_safe_mode_guard_allows_paper_bypass(monkeypatch):
    import ai_trading.execution.live_trading as live

    monkeypatch.delenv("AI_TRADING_HALT", raising=False)
    monkeypatch.setattr(live, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(live, "safe_mode_reason", lambda: "minute_gap")
    monkeypatch.setattr(live.provider_monitor, "is_disabled", lambda provider: False)
    monkeypatch.setattr(live, "_safe_mode_policy", lambda: (True, "paper"))

    blocked = live._safe_mode_guard(symbol="AAPL", side="buy", quantity=5)

    assert blocked is False
