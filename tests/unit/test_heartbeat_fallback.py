from unittest.mock import Mock

import ai_trading.data.fetch.heartbeat as hb
from ai_trading.data.fetch import fallback_order as fo


def test_fallback_called_when_primary_fails():
    """Ensure fallback provider is invoked if primary heartbeat fails."""
    fo.reset()

    primary = Mock(side_effect=RuntimeError("boom"))
    fallback = Mock(return_value="ok")

    result = hb.heartbeat(primary, fallback)

    assert result == "ok"
    assert primary.called
    assert fallback.called
    assert fo.FALLBACK_ORDER.get("yahoo")
