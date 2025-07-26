"""Load a trained RL policy and produce trade signals."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from . import RLAgent
from strategies.base import TradeSignal


def load_policy(model_path: str | Path) -> RLAgent:
    agent = RLAgent(model_path)
    agent.load()
    return agent


def predict_signal(agent: RLAgent, state: np.ndarray) -> TradeSignal | None:
    return agent.predict(state)
