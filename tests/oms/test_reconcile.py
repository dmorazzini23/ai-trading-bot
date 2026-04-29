from types import SimpleNamespace

import pytest

from ai_trading.oms.reconcile import fetch_broker_positions


def test_fetch_broker_positions_supports_alpaca_get_all_positions() -> None:
    api = SimpleNamespace(
        get_all_positions=lambda: [
            SimpleNamespace(symbol="AAPL", qty="0.5", side="long"),
            SimpleNamespace(symbol="MSFT", qty="1.25", side="short"),
        ]
    )

    positions = fetch_broker_positions(api)

    assert positions == {"AAPL": 0.5, "MSFT": -1.25}


def test_fetch_broker_positions_raises_on_fetch_failure() -> None:
    def _raise():
        raise RuntimeError("broker unavailable")

    with pytest.raises(RuntimeError, match="broker_positions_fetch_failed"):
        fetch_broker_positions(SimpleNamespace(get_all_positions=_raise))
