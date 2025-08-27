from unittest import mock
import types
import pytest


def test_no_import_time_initialization(monkeypatch):
    """Test that importing bot_engine does not initialize Alpaca clients."""
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("TRADING_MODE", "DRY_RUN")
    monkeypatch.setenv("PYTEST_RUNNING", "true")

    # Test that we can import the module without triggering initialization
    import ai_trading.core.bot_engine as eng
    # trading_client should be None since no initialization at import time
    assert eng.trading_client is None

def test_ensure_returns_tuple_and_skips_in_shadow(monkeypatch):
    """Test that _ensure_alpaca_env_or_raise always returns a tuple in SHADOW_MODE."""
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("PYTEST_RUNNING", "true")

    import ai_trading.core.bot_engine as eng
    k, s, b = eng._ensure_alpaca_env_or_raise()
    assert isinstance((k, s, b), tuple)

def test_initialize_skips_in_shadow_mode(monkeypatch):
    """Test that _initialize_alpaca_clients skips initialization in SHADOW_MODE without credentials."""
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("PYTEST_RUNNING", "true")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    import ai_trading.core.bot_engine as eng
    # Reset the client to None
    eng.trading_client = None

    # This should not raise and should skip initialization
    with mock.patch.object(eng, "logger") as mock_logger:
        eng._initialize_alpaca_clients()
        # Should have logged the skip message
        mock_logger.info.assert_any_call(
            "Shadow mode or missing credentials: skipping Alpaca client initialization"
        )
        # Client should still be None
        assert eng.trading_client is None

def test_initialize_raises_when_missing_creds_and_not_shadow(monkeypatch):
    """Test that _initialize_alpaca_clients raises when credentials missing outside SHADOW_MODE."""
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "true")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    import ai_trading.core.bot_engine as eng
    # Reset the client to None
    eng.trading_client = None

    try:
        eng._initialize_alpaca_clients()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Missing Alpaca API credentials" in str(e)


def test_ctx_api_attached_after_initialization(monkeypatch):
    """ensure_alpaca_attached attaches a trading client when creds are present."""
    pytest.importorskip("alpaca")
    eng = pytest.importorskip("ai_trading.core.bot_engine")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://example.com")
    monkeypatch.setenv("PYTEST_RUNNING", "true")

    class DummyTradingClient:
        def __init__(self, *a, **k):
            pass
    monkeypatch.setattr("alpaca.trading.client.TradingClient", DummyTradingClient)
    eng.trading_client = None

    ctx = types.SimpleNamespace()
    eng.ensure_alpaca_attached(ctx)
    assert getattr(ctx, "api", None) is not None


def test_ensure_alpaca_attached_raises_without_creds(monkeypatch):
    """ensure_alpaca_attached raises when clients cannot be initialized."""
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "true")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    import ai_trading.core.bot_engine as eng
    eng.trading_client = None

    ctx = types.SimpleNamespace()
    with pytest.raises(RuntimeError):
        eng.ensure_alpaca_attached(ctx)


def test_safe_get_account_attaches_client(monkeypatch):
    """safe_alpaca_get_account attaches a client and returns an account."""
    pytest.importorskip("alpaca")
    eng = pytest.importorskip("ai_trading.core.bot_engine")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://example.com")
    monkeypatch.setenv("PYTEST_RUNNING", "true")

    class DummyTradingClient:
        def __init__(self, *a, **k):
            pass

        def get_account(self):
            return object()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", DummyTradingClient)
    eng.trading_client = None

    ctx = types.SimpleNamespace()
    acct = eng.safe_alpaca_get_account(ctx)
    assert acct is not None
    assert getattr(ctx, "api", None) is not None
