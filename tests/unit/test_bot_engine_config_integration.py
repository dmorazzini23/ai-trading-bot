import pytest

try:
    from ai_trading.core.bot_engine import BotMode
    import ai_trading.settings as S
    import ai_trading.position_sizing as ps
except Exception as exc:  # pragma: no cover - dependency stub
    pytest.skip(f"core modules unavailable: {exc}", allow_module_level=True)


def test_botmode_config_exposes_legacy_params(monkeypatch):
    """AI-AGENT-REF: ensure BotMode seeds required legacy params."""
    monkeypatch.setattr(S, "get_capital_cap", lambda: 0.25)
    monkeypatch.setattr(S, "get_dollar_risk_limit", lambda: 0.1)
    monkeypatch.setattr(S, "get_conf_threshold", lambda: 0.9)
    monkeypatch.setattr(S, "get_kelly_fraction", lambda: 0.5, raising=False)
    monkeypatch.setattr(ps, "resolve_max_position_size", lambda capital_cap: 1000.0)

    mode = BotMode()
    params = mode.get_config()
    assert params["CAPITAL_CAP"] == 0.25
    assert params["DOLLAR_RISK_LIMIT"] == 0.1
    assert params.get("MAX_POSITION_SIZE") == 1000.0
    assert params["CONF_THRESHOLD"] == 0.9
    assert params["KELLY_FRACTION"] == 0.5
