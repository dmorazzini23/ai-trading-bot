from __future__ import annotations

import numpy as np

import ai_trading.rl_trading as rl
from ai_trading.rl_trading.env import ActionSpaceConfig
from ai_trading.rl_trading.inference import InferenceConfig, UnifiedRLInference


def test_unified_inference_handles_stub_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(rl, "is_rl_available", lambda: False)
    cfg = InferenceConfig(
        model_path=str(tmp_path / "missing.zip"),
        action_config=ActionSpaceConfig(action_type="discrete", discrete_actions=3),
    )

    inference = UnifiedRLInference(cfg)
    signal = inference.predict(np.zeros(6, dtype=np.float32), symbol="AAPL")

    assert signal is None
    stats = inference.get_stats()
    assert stats["total_predictions"] == 0
