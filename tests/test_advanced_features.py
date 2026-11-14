# Ensure repository root in path
import sys
import types
from typing import Any, cast

import pytest

pytestmark = pytest.mark.usefixtures("default_env")

# stub missing deps
req_mod = types.ModuleType("requests")
setattr(cast(Any, req_mod), "post", lambda *a, **k: None)
sys.modules.setdefault("requests", req_mod)

try:
    from ai_trading import rebalancer
except (ValueError, TypeError):
    pytest.skip("alpaca not available", allow_module_level=True)



def test_maybe_rebalance(monkeypatch):
    """maybe_rebalance triggers a rebalance after the interval."""
    calls = []
    monkeypatch.setattr(rebalancer, "rebalance_interval_min", lambda: 0)
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    rebalancer._last_rebalance = rebalancer.datetime.now(rebalancer.UTC) - rebalancer.timedelta(minutes=1)
    rebalancer.maybe_rebalance("ctx")
    assert calls == ["ctx"]






