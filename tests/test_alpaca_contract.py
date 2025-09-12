import types

from ai_trading import alpaca_api  # AI-AGENT-REF: canonical import

from tests.mocks.alpaca_mocks import MockClient


def test_submit_order_contract():
    api = MockClient()
    req = types.SimpleNamespace(symbol="AAPL", qty=1, side="buy", time_in_force="day")
    result = alpaca_api.submit_order(
        req.symbol,
        req.side,
        qty=req.qty,
        time_in_force=req.time_in_force,
        client=api,
    )
    assert result["id"] == "1"
    assert getattr(api.last_payload, "symbol", None) == "AAPL"
    assert hasattr(api.last_payload, "client_order_id")
