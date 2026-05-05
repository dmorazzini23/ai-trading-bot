from __future__ import annotations

from ai_trading.config.launch_profiles import (
    provider_authority_allows,
    resolve_launch_profile,
)


def test_resolve_launch_profile_applies_profile_scoped_overrides(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_SYMBOLS", "AAPL,NVDA")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "2")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_ALLOW_SHORTS", "0")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "12.5")

    profile = resolve_launch_profile()

    assert profile.name == "live_canary"
    assert profile.allowed_symbols == ("AAPL", "NVDA")
    assert profile.max_order_count == 2
    assert profile.max_daily_loss == 12.5
    assert profile.shorts_allowed is False
    assert profile.manual_approval_required is True


def test_provider_authority_blocks_backup_and_synthetic_quotes_for_live(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    profile = resolve_launch_profile()

    allowed, context = provider_authority_allows(
        profile=profile,
        provider_state={"active": "yahoo", "status": "healthy", "using_backup": True},
        quote_state={"source": "synthetic", "synthetic": True, "allowed": True},
        execution_mode="live",
    )

    assert allowed is False
    assert "execution_quote_not_alpaca" in context["reasons"]
    assert "backup_provider_research_only" in context["reasons"]
    assert "synthetic_quote" in context["reasons"]


def test_provider_authority_allows_healthy_alpaca_quotes_for_paper_trade(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    profile = resolve_launch_profile()

    allowed, context = provider_authority_allows(
        profile=profile,
        provider_state={"active": "alpaca-iex", "status": "healthy", "using_backup": False},
        quote_state={"source": "latest_quote", "synthetic": False, "allowed": True},
        execution_mode="paper",
    )

    assert allowed is True
    assert context["reasons"] == []
