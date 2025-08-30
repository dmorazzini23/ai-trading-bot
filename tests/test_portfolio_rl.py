import numpy as np
import pytest

from ai_trading.utils.device import TORCH_AVAILABLE
if not TORCH_AVAILABLE:
    pytest.skip("torch not installed", allow_module_level=True)
import torch


def test_rebalance_portfolio_normalizes_weights():
    from ai_trading.portfolio_rl import PortfolioReinforcementLearner

    learner = PortfolioReinforcementLearner()
    state = np.random.rand(learner.state_dim)
    weights = learner.rebalance_portfolio(state)
    assert weights.shape[0] == learner.action_dim
    assert isinstance(learner.actor.net[0], torch.nn.Linear)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)


def test_import_error_when_torch_missing(monkeypatch):
    import builtins
    import sys
    import ai_trading.portfolio_rl as prl

    prl._lazy_import_torch.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torch"):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="PyTorch is required for ai_trading.portfolio_rl"):
        prl.PortfolioReinforcementLearner()

