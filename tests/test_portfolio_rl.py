import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_rebalance_portfolio_normalizes_weights():
    from ai_trading.portfolio_rl import PortfolioReinforcementLearner

    learner = PortfolioReinforcementLearner()
    state = np.random.rand(learner.state_dim)
    weights = learner.rebalance_portfolio(state)
    assert weights.shape[0] == learner.action_dim
    assert isinstance(learner.actor.net[0], torch.nn.Linear)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

