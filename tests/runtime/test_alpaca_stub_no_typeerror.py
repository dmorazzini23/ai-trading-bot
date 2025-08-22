
def test_alpaca_stub_accepts_args_and_noops():
    # Import from bot_engine where the stub is defined/aliased
    from ai_trading.core.bot_engine import (
        _AlpacaStub,  # private but stable for this test
    )
    stub = _AlpacaStub("key", "secret", base_url="https://example.com", paper=True)
    # Should not raise; attribute/method access should no-op
    assert stub is not None
    assert stub.any_method(1, x=2) is None
