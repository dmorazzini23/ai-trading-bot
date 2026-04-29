import types
from datetime import UTC, datetime, timedelta

import pytest


def test_get_bars_raises_when_settings_missing(monkeypatch):
    from ai_trading.data import fetch

    monkeypatch.setattr(fetch, "_load_settings", lambda: None)

    start = datetime.now(UTC) - timedelta(minutes=1)
    end = datetime.now(UTC)
    with pytest.raises(RuntimeError):
        fetch.get_bars("AAPL", "1Min", start, end)


def test_main_exits_when_env_invalid(monkeypatch):
    import ai_trading.main as m

    def bad_validate():
        raise RuntimeError("missing env")

    monkeypatch.setattr(m, "validate_required_env", lambda *a, **k: bad_validate())
    with pytest.raises(SystemExit) as excinfo:
        m.main([])
    assert excinfo.value.code == 1


def test_fail_fast_env_backfills_positive_risk_defaults(monkeypatch):
    import ai_trading.main as m

    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_DATA_FEED",
        "WEBHOOK_SECRET",
        "AI_TRADING_CAPITAL_CAP",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
        "ALPACA_API_URL",
        "ALPACA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    class DummyConfig:
        @classmethod
        def from_env(cls, *args, **kwargs):  # noqa: D401, ARG003
            return cls()

    monkeypatch.setattr(m, "_trading_config_from_env", DummyConfig.from_env)
    monkeypatch.setattr(m, "reload_env", lambda override=False: None)
    monkeypatch.setattr(m, "validate_required_env", lambda *a, **k: ())
    monkeypatch.setattr(m, "redact_config_env", lambda snapshot: snapshot)
    monkeypatch.setattr(
        m,
        "logger",
        types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, critical=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        m,
        "_resolve_alpaca_env",
        lambda: ("test-key", "test-secret", "https://paper-api.alpaca.markets"),
    )

    m._fail_fast_env()

    from ai_trading.config.management import get_env

    assert str(get_env("AI_TRADING_CAPITAL_CAP")) == "0.25"
    assert str(get_env("DOLLAR_RISK_LIMIT")) == "0.05"


def test_fail_fast_env_requires_alpaca_credentials_in_main_runtime(monkeypatch):
    import ai_trading.main as m

    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_DATA_FEED",
        "WEBHOOK_SECRET",
        "AI_TRADING_CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
        "RUN_HEALTHCHECK",
        "SHADOW_MODE",
        "PYTEST_RUNNING",
        "TESTING",
        "PYTEST_CURRENT_TEST",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(m, "_is_test_mode", lambda: False)
    monkeypatch.setattr(m, "_trading_config_from_env", lambda **_kwargs: object())
    monkeypatch.setattr(m, "reload_env", lambda override=False: None)
    monkeypatch.setattr(m, "validate_no_deprecated_env", lambda: None)
    monkeypatch.setattr(
        m,
        "validate_required_env",
        lambda _keys: (_ for _ in ()).throw(
            RuntimeError("Missing required environment variable(s): ALPACA_API_KEY, ALPACA_SECRET_KEY")
        ),
    )
    monkeypatch.setattr(m, "_log_config_effective_summary", lambda _cfg: None)
    monkeypatch.setattr(m, "_resolve_alpaca_env", lambda: (None, None, "https://paper-api.alpaca.markets"))

    with pytest.raises(SystemExit) as excinfo:
        m._fail_fast_env()

    assert excinfo.value.code == 1


def test_fail_fast_env_tolerates_missing_alpaca_credentials_in_shadow(monkeypatch):
    import ai_trading.main as m

    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "PYTEST_RUNNING",
        "TESTING",
        "PYTEST_CURRENT_TEST",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHADOW_MODE", "1")

    observed_required: list[tuple[str, ...]] = []

    monkeypatch.setattr(m, "_is_test_mode", lambda: False)
    monkeypatch.setattr(m, "_trading_config_from_env", lambda **_kwargs: object())
    monkeypatch.setattr(m, "reload_env", lambda override=False: None)
    monkeypatch.setattr(m, "validate_no_deprecated_env", lambda: None)
    monkeypatch.setattr(m, "redact_config_env", lambda snapshot: snapshot)
    monkeypatch.setattr(m, "_log_config_effective_summary", lambda _cfg: None)
    monkeypatch.setattr(m, "_resolve_alpaca_env", lambda: (None, None, "https://paper-api.alpaca.markets"))
    monkeypatch.setattr(m, "_preflight_rl_runtime_model_path_permissions", lambda: None)

    def fake_validate_required(keys):
        observed_required.append(tuple(keys))
        return {}

    monkeypatch.setattr(m, "validate_required_env", fake_validate_required)
    monkeypatch.setattr(
        m,
        "logger",
        types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, critical=lambda *a, **k: None),
    )

    m._fail_fast_env()

    assert observed_required
    assert "ALPACA_API_KEY" not in observed_required[-1]
    assert "ALPACA_SECRET_KEY" not in observed_required[-1]
