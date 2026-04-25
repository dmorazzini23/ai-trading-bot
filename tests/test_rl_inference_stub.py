from __future__ import annotations

import numpy as np
import pytest

from ai_trading.rl_trading.env import ActionSpaceConfig
from ai_trading.rl_trading import inference as inference_mod
from ai_trading.rl_trading.inference import InferenceConfig, UnifiedRLInference


def test_unified_inference_handles_stub_agent(monkeypatch, tmp_path):
    class _UnavailableAgent:
        def __init__(self, model_path):
            self.model_path = model_path

        def load(self):
            raise ImportError("RL stack not available")

    monkeypatch.setattr(inference_mod, "RLAgent", _UnavailableAgent)
    cfg = InferenceConfig(
        model_path=str(tmp_path / "missing.zip"),
        action_config=ActionSpaceConfig(action_type="discrete", discrete_actions=3),
    )

    with pytest.raises(ImportError):
        UnifiedRLInference(cfg)
