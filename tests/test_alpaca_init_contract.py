from unittest import mock

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
        mock_logger.info.assert_called_with("Shadow mode or missing credentials: skipping Alpaca client initialization")
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